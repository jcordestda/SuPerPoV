# Daily calculation caching and persistence benchmarks

The application now uses `superpov_compute.DailyAnalyzer` for its plotted field
and daily scores. The original `compute_data` function and its helpers remain
unchanged as a numerical and timing reference. These tools measure complete
calculation calls as well as the individual topology improvements.

## Complete-call benchmark

Three repetitions per date, reporting medians for complete calculation calls:

| Date | Original | First query, fresh analyzer | New date, cached analyzer | First-query speedup | Cached-query speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1965-11-01 | 2.129 s | 1.086 s | 78.8 ms | 1.96x | 27.0x |
| 1965-12-23 | 4.218 s | 1.089 s | 81.1 ms | 3.87x | 52.0x |
| 1966-04-30 | 9.285 s | 1.088 s | 80.9 ms | 8.53x | 114.7x |

The later-date reference calls spend longer scanning rows; indexed access removes
that growth. For the December example, first-query elapsed time fell by about 74%
and cached-query time by about 98%. All measurements are CPU calculations.
These are complete calculation-call timings, not measurements of GUI rendering
or launching a new Python process.

The integrated engine also passed exact checks on all 181 days: 362 diagrams,
17,476 positive-persistence bars, complete ordered depth lists, both scores,
and the selected matching bar. Maximum observed endpoint and depth error was
zero. Both timed candidates matched the original function on every repetition.
The [reference report](results/daily_reference_results.json) contains source and
input hashes, raw repeated timings, the timing boundaries, and per-day checks.
Timing was collected separately after the regression suites completed; the
whole-winter validation used identical source hashes and input.

`compare_daily.py` measures the original calculation, the first query on a fresh
analyzer, and a different-date query after the analyzer has prepared another day.
The fresh-query timer includes analyzer construction, file selection/header checks, parsing,
input validation, date indexing, triangulation, template preparation, and scores.
The cached-query timer includes file fingerprint checks and a new persistence
calculation. Completed daily results are never cached.

Only the requested winter files are fully parsed; other candidate files need a
one-row header probe. Date lookup then uses a dictionary and a NumPy slice.
Each analyzer retains at most eight winter arrays (about 96 MB of height values
at this resolution for eight 181-day winters), plus its fixed geometry templates
and bounded header metadata. It releases parsed dataframes. File identity, size,
modification time, and change time determine whether a file needs to be reloaded.
Returned arrays and results can be modified without changing cached data.

The GUI shares one analyzer across plotted data, selected pressures, and dates.
Run several dates through the headless CLI in one process to get the same reuse.
Each new process pays its own preparation cost. The benchmark measures warm
filesystem caches; imports, process startup, GUI rendering, wind calculations,
and output serialization are excluded from both reference and candidate timing.
The recorded runs supply one input file; GUI directory discovery is excluded,
and selecting a winter from a larger archive adds header and file-stat work.

```bash
benchmarks/.venv/bin/python benchmarks/compare_daily.py \
  --file gph_winter_1965.txt --repeats 3 --validate-all
```

Every timed output must exactly equal the original function's complete result.
The all-day check compares both full diagrams and all score details against
the existing extended construction, and checks that this reference reproduces
the original function on the timed dates. `--validation-only` skips timing and
the original-function bridge; `--method extended` selects the extended candidate.

Input validation rejects unsupported grid layouts and mixed pressure levels.
It supports missing calendar days and the final date in a partial winter when
that date is present; these cases previously failed in the scanning date helper.
The GUI's date-window helpers and wind-data path are outside this change.

## Earlier topology-only experiment

On the supplied 1965–1966 winter, reusing the fixed simplex structure and
computing ordinary persistence on negated heights reduced the time for both
surfaces from approximately **424 ms to 82 ms**. That is about **5.2 times faster
for the topology calculation**, with data and geometry already prepared. It is
not a measurement of application startup or an A100 speedup: all computation in
these experiments runs on the CPU.

## What the original profile measures

`profile_cpu.py` extracts `correct_df`, `date_index_returner`, and `compute_data`
from `SuPerPoV3.1.0.py` using Python's AST. It executes their original statements
without importing the GUI or executing the application's main section. A second
copy wraps contiguous statements in timers and verifies that its complete output
equals the original output. It rejects source changes that invalidate the stage
mapping.

For 1965-12-23, the median original call took **4.264 s** over three repetitions:

| Stage | Median time |
| --- | ---: |
| Locate the day by scanning the dataframe | 2.134 s |
| Read and parse the matching winter file twice | 1.316 s |
| Construct both Delaunay triangulations | 0.244 s |
| Construct simplex trees and assign heights | 0.209 s |
| Extend both filtrations | 0.093 s |
| Compute extended persistence | 0.135 s |
| Extract the daily height array | 0.081 s |

Stage medians need not sum to the median total. Filtration repair, cleanup,
and small bookkeeping costs are additional. The original median was 2.150 s
on 1965-11-01 and 9.465 s on 1966-04-30; the increasing row scan accounts for
most of this date dependence.

The input is explicitly read before timing, so these are **warm filesystem-cache
measurements**. Original parsing and scanning still happen on every call. Only
one candidate winter file is supplied; searching a larger data directory can
cost more. Imports, GUI interaction, plotting, optional wind data, and application
startup are excluded. Optional cProfile calls are separate and are not used for
the wall-clock speed claims because cProfile adds substantial overhead.

## Why the faster calculation preserves the requested features

Section 2.2 of the [SuPerPoV paper by Cordes, Giunti, and Wu](https://arxiv.org/html/2601.11386v1#S2.SS2)
describes scanning geopotential height downward and measuring the lifetimes of
one-dimensional holes. These are superlevel-set H1 features.

The current implementation computes extended persistence of the height function
`f`, then uses only the relative H2 bars. The EP Symmetry Corollary in
[Carlsson, de Silva, and Morozov, *Zigzag Persistent Homology and Real-valued Functions*, §3](https://www.mrzv.org/publications/zigzags/socg09/)
identifies these finite bars with ordinary H1 bars of `-f`, reflecting both
endpoint coordinates. The [GUDHI extended-persistence tutorial, equation (6)](https://github.com/GUDHI/TDA-tutorial/blob/master/tutorials/Tuto-GUDHI-extended-persistence.ipynb)
states the same relationship.

The direct variant therefore:

1. Uses exactly the same vertices and Delaunay triangles as the current code.
2. Assigns each vertex its negated geopotential height and gives each higher
   simplex the maximum value of its vertices.
3. Computes ordinary persistence and extracts finite H1 intervals.
4. Negates both interval endpoints and applies the current scoring rules.

This avoids building the extended filtration and computing its additional
diagrams. It retains coefficient field 11, the `1e-5` persistence cutoff, and
double precision. In particular, displacement matching still uses the first
grid bar with a death value exactly equal to the death of the longest polar
bar. It does not substitute a different matching rule.

Both original geometries are preserved. The polar triangulation includes its
filled central cap and longitude seam; the second triangulation is the open
grid made by duplicating the field horizontally. Replacing them with an arbitrary
grid adjacency or a different triangulation is a separate scientific change.

## Reusing the simplex structure safely

The topology is fixed for a given resolution. `fast_build.py` prepares a pristine
simplex-tree template once. Its vertices initially have `+inf` filtration and
its edges and triangles have `-inf` filtration. Each daily calculation copies
this template, inserts all finite vertex heights in one batch, and repairs the
filtration.

The asymmetric infinities are deliberate: GUDHI insertion can lower an existing
filtration, while `make_filtration_non_decreasing` can only raise it. Starting
vertices at `+inf` permits the batch update; starting higher simplices at `-inf`
permits the repair pass to establish the correct vertex maximum. The intermediate
template is not a valid filtration and must never be used for persistence itself.

A tree already processed for another day is not the template. Reassigning only
its vertices can leave stale edge and triangle values, and extended persistence
also modifies the tree. Each daily computation receives a fresh copy.

## Measurements and validation

Five repetitions, rotating the execution order among four methods, produced:

| Date | Existing construction + extended | Direct H1 | Template + extended | Template + direct H1 |
| --- | ---: | ---: | ---: | ---: |
| 1965-11-01 | 421.4 ms | 253.5 ms | 239.6 ms | 79.9 ms |
| 1965-12-23 | 423.7 ms | 254.3 ms | 246.5 ms | 81.7 ms |
| 1966-04-30 | 422.3 ms | 255.0 ms | 242.5 ms | 82.2 ms |

These times include creating or copying both daily trees, assigning vertices,
repairing filtrations, persistence, extracting and sorting bars, and scoring.
They exclude TSV parsing, extracting daily arrays from the dataframe, Delaunay triangulation,
template preparation, and imports. The one-time geometry cost was approximately
0.27 s and template preparation approximately 0.03 s. They must be included when
evaluating a fresh process or a first query.

The reference environment used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
pandas 2.3.3, and GUDHI 3.13.0 on a Linux VM reporting 12 logical CPUs. These are
observations on this machine and dataset, not general performance guarantees.

Validation compared the existing extended construction with template + direct
H1 for all **181 days, 362 diagrams, and 17,476 positive-persistence bars**.
Every endpoint, ordered depth list, split score, displacement score, and matched
bar was numerically equal, with maximum observed endpoint and depth error zero.
Here "exact" means floating-point value equality, not a comparison of serialized
bit patterns. The reference computation's scores and leading depths were also
checked against the actual original function on the three benchmark dates.

The data file contains daily 10 hPa fields from 1965-11-01 through 1966-04-30:
46 latitudes ordered from 90 to 0 by -2 degrees and 180 longitudes ordered from
0 to 358 by 2 degrees. It has 1,498,680 rows. The input hash, recorded timings,
and per-day validation summary are in [results/reference_results.json](results/reference_results.json).
The 54.5 MB input and generated profiler binaries are not included.

Exact agreement on this winter is not a proof of bitwise agreement on every
input or GUDHI version. GUDHI documents possible rounding during the scaling
used by extended persistence. Direct persistence is now the default. Other
resolutions, pressure levels, tied values, dependency versions, or triangulations
deserve their own numerical checks; exact equality on them is not assumed.
Use `--method extended` in the headless CLI or daily benchmark for comparisons.

The current source is the reference for this comparison. Its April 30
displacement score already differs slightly from the repository's historical
precomputed table: 0.6582677108 versus 0.6584103553. The faster prototype reproduces
the current source value. This experiment does not establish the cause of the
historical difference.

## Reproduce

Use Python 3.10 to reproduce the recorded dependency environment:

```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/python -m pip install -r benchmarks/requirements.txt
```

Obtain `gph_winter_1965.txt` from the data associated with the project; the main
[README](../README.md) links the [Zenodo dataset](https://doi.org/10.5281/zenodo.17903131).
Pass its path explicitly if it is outside the repository root.

Profile the original calculation:

```bash
benchmarks/.venv/bin/python benchmarks/profile_cpu.py \
  --file gph_winter_1965.txt \
  --dates 1965-11-01 1965-12-23 1966-04-30 \
  --repeats 3 \
  --output benchmarks/results/cpu_profile.json \
  --cprofile-dir benchmarks/results/cprofile
```

Compare all four persistence implementations:

```bash
benchmarks/.venv/bin/python benchmarks/compare_persistence.py \
  --file gph_winter_1965.txt --repeats 5
```

Validate every day against the existing extended calculation:

```bash
benchmarks/.venv/bin/python benchmarks/compare_persistence.py \
  --file gph_winter_1965.txt --all --repeats 1 \
  --methods extended template_direct --require-exact
```

The one-repetition whole-winter run is a correctness check; use the repeated
three-date run for the timing comparison. Generated results are ignored by Git,
except for the deliberately checked-in compact reference report.

Input checks run outside the timed calculations. They reject unsupported grid
layouts, nonfinite values, missing dates, and output paths that would overwrite
the input or Python source. Comparison runs validate every repetition and report
numeric exactness separately from the configured numerical tolerances.
`--require-exact` makes exact numeric equality a requirement for a successful
exit. Validation remains enabled with `python -O`.

Run the focused regression checks with:

```bash
benchmarks/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
benchmarks/.venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py'
```

The application tests exercise cache invalidation, bounded file retention,
pressure isolation, shared plotting/calculation reads, CLI output, and agreement
with the original calculation on small fields. The benchmark tests verify that
fresh timers include construction, cached queries use a different seed date,
every repetition is checked, and full grid diagrams are compared even when
scores happen to agree. Both suites also run with `python -O`.
