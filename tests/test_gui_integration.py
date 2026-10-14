"""Exercise the application's cache wiring without importing Tk or Matplotlib."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'SuPerPoV3.1.0.py'


class GuiCacheIntegration(unittest.TestCase):
    def main_fragments(self):
        nodes = ast.parse(SOURCE.read_text()).body
        starts = [index for index, node in enumerate(nodes)
                  if isinstance(node, ast.ImportFrom) and node.module == 'superpov_compute']
        self.assertEqual(len(starts), 1, 'GUI setup must import the cached analyzer once')
        start = starts[0]
        stop = next(index for index in range(start + 1, len(nodes))
                    if isinstance(nodes[index], ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'Z'
                            for target in nodes[index].targets))
        daily_loop = next(node for node in nodes
                          if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
                          and node.iter.id == 'sorted_dates')
        return nodes[start:stop + 1], [daily_loop]

    def execute(self, statements, namespace):
        module = ast.Module(body=statements, type_ignores=[])
        exec(compile(module, str(SOURCE), 'exec'), namespace)

    def test_plot_and_daily_results_share_one_cache_and_route_selected_pressures(self):
        setup, daily = self.main_fragments()
        from superpov_compute import DailyAnalyzer
        import superpov_compute

        resolution = (3, 4)
        dates = ['1965-11-01', '1965-11-02']
        with tempfile.TemporaryDirectory() as directory:
            files = {}
            for pressure in (2, 5, 10, 50, 100):
                path = Path(directory) / f'{pressure}.tsv'
                frame = pd.DataFrame({
                    '# date': np.repeat(dates, 12),
                    'lev': np.full(24, pressure),
                    'lat': np.tile(np.repeat([90., 45., 0.], 4), 2),
                    'lon': np.tile([0., 90., 180., 270.], 6),
                    'data': np.tile(np.repeat([100., 200., 300.], 4), 2) + pressure,
                })
                frame.to_csv(path, sep='\t', index=False)
                files[pressure] = [str(path)]

            for selected in ([2, 5, 10, 50, 100], [5, 50]):
                with self.subTest(selected=selected):
                    instances = []

                    def create_analyzer(*args, **kwargs):
                        kwargs['cache_dir'] = Path(directory) / f'cache-{len(selected)}'
                        analyzer = DailyAnalyzer(*args, **kwargs)
                        analyzer.get_values = Mock(wraps=analyzer.get_values)
                        analyzer.compute_data = Mock(wraps=analyzer.compute_data)
                        instances.append(analyzer)
                        return analyzer

                    namespace = {'np': np, 'date': dates[0], 'resolution': resolution,
                                 'plot_type': 50, 'pressures': selected, 'sorted_dates': dates}
                    namespace.update({f'txt_files{pressure}': paths for pressure, paths in files.items()})
                    namespace.update({f'all_data{pressure}': [] for pressure in files})
                    with patch.object(superpov_compute, 'DailyAnalyzer', side_effect=create_analyzer), \
                            patch.object(pd, 'read_csv', wraps=pd.read_csv) as reads:
                        self.execute(setup, namespace)
                        self.assertEqual(len(instances), 1)
                        analyzer = instances[0]
                        analyzer.get_values.assert_called_once_with(dates[0], files[50], pressure=50)
                        np.testing.assert_array_equal(namespace['Z'],
                                                      [[150.] * 4, [250.] * 4, [350.] * 4])
                        self.execute(daily, namespace)
                    self.assertEqual(len(instances), 1)
                    self.assertEqual(analyzer.compute_data.call_args_list,
                                     [call(date, files[pressure], pressure=pressure)
                                      for date in dates for pressure in selected])
                    for pressure in files:
                        self.assertEqual(len(namespace[f'all_data{pressure}']),
                                         len(dates) if pressure in selected else 0)
                    # Header probes may occur separately; each selected winter is
                    # fully parsed once across plotting and both daily computations.
                    full_reads = [item for item in reads.call_args_list
                                  if item.kwargs.get('nrows') is None]
                    self.assertEqual(len(full_reads), len(selected))


class GuiWindIntegration(unittest.TestCase):
    def execute_wind_branch(self, enabled):
        nodes = ast.parse(SOURCE.read_text()).body
        start = next(index for index, node in enumerate(nodes)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'neg_wind_dates'
                             for target in node.targets))
        stop = next(index for index in range(start, len(nodes))
                    if isinstance(nodes[index], ast.If)
                    and isinstance(nodes[index].test, ast.Compare)
                    and isinstance(nodes[index].test.left, ast.Name)
                    and nodes[index].test.left.id == 'wind_yn')
        namespace = {
            'pd': pd, 'wind_yn': 'E' if enabled else 'F',
            'sorted_dates': ['1965-11-01', '1965-11-02', '1965-11-03'],
            'input_dict': {'WindData': pd.DataFrame({
                'DATES': ['1965-10-31', '1965-11-01', '1965-11-02', '1965-11-03'],
                'SPEEDS': [-1., -2., 0., 3.],
            })},
        }
        namespace.update({f'{ratio}_ratios{pressure}': [.25, .5, .75] if pressure in (10, 50) else []
                          for ratio in ('disp', 'split') for pressure in (2, 5, 10, 50, 100)})
        with patch.object(pd, 'read_csv', side_effect=AssertionError(
                'Plotting must reuse the wind data validated before closing the form')):
            exec(compile(ast.Module(body=nodes[start:stop + 1], type_ignores=[]),
                         str(SOURCE), 'exec'), namespace)
        return namespace

    def test_enabled_wind_reuses_validated_data_and_preserves_reversal_markers(self):
        namespace = self.execute_wind_branch(True)
        self.assertEqual(namespace['neg_wind_dates'], ['1965-11-01'])
        for pressure in (2, 5, 10, 50, 100):
            for prefix in ('NWD_D_data', 'NWD_S_data'):
                self.assertEqual(namespace[f'{prefix}{pressure}'],
                                 [.25] if pressure in (10, 50) else [])

    def test_disabled_wind_has_no_markers_or_file_reads(self):
        namespace = self.execute_wind_branch(False)
        self.assertEqual(namespace['neg_wind_dates'], [])
        for pressure in (2, 5, 10, 50, 100):
            self.assertEqual(namespace[f'NWD_D_data{pressure}'], [])
            self.assertEqual(namespace[f'NWD_S_data{pressure}'], [])


if __name__ == '__main__':
    unittest.main()
