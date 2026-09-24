"""Fetch race schedules, qualifying results, and race results per season/round.

Pulls from the Jolpica Ergast-compatible API (see f1_predict.clients.ergast)
and caches every raw response to disk. Only produces raw, per-round data
structures — reshaping into a driver/race dataset happens in build_dataset.py.
"""

import asyncio
import datetime
from dataclasses import dataclass, field

from f1_predict.clients.ergast import ergast_get
from f1_predict.clients.http import close_client

START_SEASON = 2022


def current_season() -> int:
    """The latest season to ingest, based on today's date."""
    return datetime.date.today().year


@dataclass
class RoundData:
    season: int
    round: int
    race_name: str
    date: str
    circuit_id: str
    circuit_name: str
    country: str
    qualifying: dict | None = None
    results: dict | None = None
    sprint: dict | None = None
    fetch_error: str | None = None


@dataclass
class IngestResult:
    rounds: list[RoundData] = field(default_factory=list)
    failures: list[tuple[int, int, str]] = field(default_factory=list)


async def fetch_season_schedule(season: int) -> list[dict]:
    """Return the race calendar entries for a season, oldest round first."""
    data = await ergast_get(f"{season}")
    races = data["MRData"]["RaceTable"].get("Races", [])
    return races


async def fetch_round(season: int, round_no: int) -> tuple[dict | None, dict | None, dict | None]:
    """Fetch qualifying + race results + sprint results (if any) for one round.

    Sprint results are known before the main race (sprint runs the day
    before), so they're a legitimate pre-race feature on that weekend's
    main-race rows, not a leakage risk.

    Quali + results are fetched in parallel first. Sprint is fetched after,
    and only once we know the round actually happened (results is non-empty)
    — that lets us permanently cache "no sprint this weekend" for completed
    non-sprint rounds instead of re-querying it on every ingestion re-run.
    """
    quali_data, results_data = await asyncio.gather(
        ergast_get(f"{season}/{round_no}/qualifying"),
        ergast_get(f"{season}/{round_no}/results"),
    )
    quali_races = quali_data["MRData"]["RaceTable"].get("Races", [])
    results_races = results_data["MRData"]["RaceTable"].get("Races", [])
    quali = quali_races[0] if quali_races else None
    results = results_races[0] if results_races else None

    sprint = None
    if results is not None:
        sprint_data = await ergast_get(f"{season}/{round_no}/sprint", cache_empty=True)
        sprint_races = sprint_data["MRData"]["RaceTable"].get("Races", [])
        sprint = sprint_races[0] if sprint_races else None

    return quali, results, sprint


async def ingest_all(start_season: int = START_SEASON, end_season: int | None = None) -> IngestResult:
    """Ingest every completed round from start_season through end_season.

    A round is "completed" if the results endpoint returns a non-empty
    Results list — that naturally excludes future rounds on the current
    season's calendar without needing date-cutoff logic.
    """
    end_season = end_season if end_season is not None else current_season()
    result = IngestResult()
    today = datetime.date.today()

    for season in range(start_season, end_season + 1):
        try:
            schedule = await fetch_season_schedule(season)
        except Exception as exc:
            result.failures.append((season, 0, f"schedule fetch failed: {exc}"))
            continue

        for race in schedule:
            round_no = int(race["round"])
            race_date = race.get("date", "")
            # Skip rounds that clearly haven't happened yet — avoids a
            # pointless network round-trip for the rest of a future calendar.
            if race_date and race_date > today.isoformat():
                continue

            circuit = race.get("Circuit", {})
            location = circuit.get("Location", {})
            round_data = RoundData(
                season=season,
                round=round_no,
                race_name=race.get("raceName", ""),
                date=race_date,
                circuit_id=circuit.get("circuitId", ""),
                circuit_name=circuit.get("circuitName", ""),
                country=location.get("country", ""),
            )
            try:
                quali, results, sprint = await fetch_round(season, round_no)
            except Exception as exc:
                round_data.fetch_error = str(exc)
                result.failures.append((season, round_no, str(exc)))
                result.rounds.append(round_data)
                continue

            if results is None:
                # Results endpoint came back empty — race hasn't been
                # classified yet (e.g. happening today), not a real failure.
                continue

            round_data.qualifying = quali
            round_data.results = results
            round_data.sprint = sprint
            result.rounds.append(round_data)

    return result


async def _main() -> IngestResult:
    try:
        return await ingest_all()
    finally:
        await close_client()


def run() -> IngestResult:
    return asyncio.run(_main())


if __name__ == "__main__":
    ingest_result = run()
    print(f"Ingested {len(ingest_result.rounds)} rounds")
    if ingest_result.failures:
        print(f"Failures: {ingest_result.failures}")
