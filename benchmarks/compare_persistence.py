"""Experimental persistence benchmark; the application is not imported or edited.

Examples (run from the repository root with NumPy, SciPy, pandas, GUDHI):
    python benchmarks/compare_persistence.py --repeats 5
    python benchmarks/compare_persistence.py --all --repeats 1 \
        --methods extended template_direct

The four candidates preserve the current Delaunay triangles. Direct methods
replace relative H2 from extended persistence of f with finite ordinary H1
of -f, using the central reflection of endpoint coordinates described by
Carlsson, de Silva, Morozov, EP Symmetry Corollary (2009):
https://www.mrzv.org/publications/zigzags/socg09/

Comparisons preserve the application's exact death-value matching for scores.
Times exclude file reads, Delaunay, and reusable-template preparation, each
reported separately. They are topology-core timings, not full GUI latency.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import Delaunay
import gudhi as gd
from fast_build import prepare_triangle_template
from input_validation import validate_frame, validate_output_path

DATA = Path(__file__).resolve().parents[1] / 'gph_winter_1965.txt'
RES = (46, 180)
EPS = 1e-5
METHODS = ('extended','direct','template_extended','template_direct')

def geometry():
    r, theta = np.meshgrid(np.linspace(.001, 4, RES[0]), np.linspace(0, 2*np.pi, RES[1], endpoint=False), indexing='ij')
    polar = np.column_stack(((r*np.cos(theta)).ravel(), (r*np.sin(theta)).ravel()))
    x,y = np.meshgrid(np.linspace(0,90,RES[0]), np.linspace(0,720,2*RES[1],endpoint=False), indexing='ij')
    grid = np.column_stack((x.ravel(),y.ravel()))
    return [Delaunay(points).simplices for points in (polar,grid)]

def pairs(triangles, values, method, template=None, phases=None):
    is_extended=method.endswith('extended')
    phase_data={}
    t=time.perf_counter()
    if method.startswith('template_'):
        pristine,vertices=template
        st=pristine.copy()
        vertex_values=np.asarray(values,dtype=np.float64) if is_extended else -np.asarray(values,dtype=np.float64)
        st.insert_batch(vertices,vertex_values[vertices[0]])
    else:
        st=gd.SimplexTree()
        for triangle in triangles:
            st.insert([v for v in triangle], -10. if is_extended else -np.inf)
        for i,value in enumerate(values):
            if st.find([i]):
                st.assign_filtration([i], float(value) if is_extended else -float(value))
    phase_data['build_and_assign_s']=time.perf_counter()-t
    t=time.perf_counter()
    st.make_filtration_non_decreasing()
    phase_data['make_filtration_s']=time.perf_counter()-t
    if is_extended:
        t=time.perf_counter()
        st.extend_filtration()
        phase_data['extend_filtration_s']=time.perf_counter()-t
        t=time.perf_counter()
        bars=[pair for dim,pair in st.extended_persistence(min_persistence=EPS)[1] if dim==2]
    else:
        phase_data['extend_filtration_s']=0.
        t=time.perf_counter()
        st.compute_persistence(homology_coeff_field=11,min_persistence=EPS,persistence_dim_max=False)
        intervals=st.persistence_intervals_in_dimension(1)
        if not np.isfinite(intervals).all():
            raise ValueError('Direct persistence returned nonfinite intervals')
        bars=[tuple(-p) for p in intervals]
    phase_data['persistence_and_extract_s']=time.perf_counter()-t
    t=time.perf_counter()
    bars=sorted(bars,key=lambda p:p[0]-p[1],reverse=True)
    phase_data['sort_s']=time.perf_counter()-t
    if phases is not None:
        for key,value in phase_data.items():
            phases[key]=phases.get(key,0.)+value
    return bars

def scores(bars):
    cyl,grid=bars
    if not cyl:
        return {'empty_cyl':True,'depths':[],'split':None,'disp':None,'matched':None}
    depths=[birth-death for birth,death in cyl]
    split=(depths[1]/depths[0]) if len(cyl)>1 else 0.
    # Exact match and first matching grid bar deliberately preserve current code.
    matched=next((p for p in grid if p[1]==cyl[0][1]),None)
    disp=0. if matched is None else (matched[0]-matched[1])/depths[0]
    return {'depths':depths,'split':split,'disp':disp,'matched':matched}

def calculate(tris, z, method, templates=None, phases=None):
    values=[z,np.hstack((z.reshape(RES),z.reshape(RES))).ravel()]
    if templates is None:
        templates=[None,None]
    bars=[pairs(t,v,method,template,phases) for t,v,template in zip(tris,values,templates)]
    return bars,scores(bars)

def compare(ref, actual):
    rb,rs=ref; ab,as_=actual
    report={'bar_counts':[len(b) for b in rb], 'endpoints_max_abs':[], 'diagram_within_tolerance':True,'diagram_exact':True}
    for r,a in zip(rb,ab):
        r=np.asarray(sorted(r)).reshape(-1,2); a=np.asarray(sorted(a)).reshape(-1,2)
        if r.shape!=a.shape:
            report['diagram_within_tolerance']=False
            report['diagram_exact']=False
            report['endpoints_max_abs'].append(None)
        else:
            report['endpoints_max_abs'].append(float(np.max(np.abs(r-a),initial=0)))
            report['diagram_within_tolerance'] &= bool(np.allclose(r,a,rtol=0,atol=1e-8))
            report['diagram_exact'] &= bool(np.array_equal(r,a))
    report['reference_scores']={k:v for k,v in rs.items() if k!='depths'}
    report['candidate_scores']={k:v for k,v in as_.items() if k!='depths'}
    report['top_depths_reference']=rs['depths'][:2]
    report['top_depths_candidate']=as_['depths'][:2]
    report['scores_exact']=rs==as_
    report['depths_list_exact']=rs['depths']==as_['depths']
    report['depths_max_abs']=(float(np.max(np.abs(np.asarray(rs['depths'])-np.asarray(as_['depths'])),initial=0))
                              if len(rs['depths'])==len(as_['depths']) else None)
    if rs.get('empty_cyl') or as_.get('empty_cyl'):
        report['scores_within_tolerance']=rs.get('empty_cyl')==as_.get('empty_cyl')
    else:
        report['scores_within_tolerance']=bool(np.allclose([rs['split'],rs['disp']],[as_['split'],as_['disp']],rtol=0,atol=1e-10))
    return report


def verification_status(comparisons, require_exact):
    within_tolerance = all(c['diagram_within_tolerance'] and c['scores_within_tolerance'] for c in comparisons)
    exact = all(c['diagram_exact'] and c['scores_exact'] for c in comparisons)
    return {'within_tolerance': within_tolerance, 'exact': exact,
            'requirement': 'exact' if require_exact else 'tolerance',
            'passed': exact if require_exact else within_tolerance}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=Path,default=DATA)
    parser.add_argument('--output',type=Path)
    dates_group=parser.add_mutually_exclusive_group()
    dates_group.add_argument('--all',action='store_true')
    dates_group.add_argument('--dates',nargs='+',help='Explicit dates; default: three 1965/66 example dates')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--require-exact',action='store_true',help='Fail on any numeric difference, even within reported tolerances')
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least 1')
    selected_methods=tuple(dict.fromkeys(args.methods))
    if 'extended' not in selected_methods:
        parser.error('--methods must include the extended reference')
    if not args.file.is_file():
        parser.error(f'Input file does not exist: {args.file}')
    destination=args.output or Path(__file__).resolve().parent / 'results' / ('persistence_all_days.json' if args.all else 'persistence_comparison.json')
    try:
        destination=validate_output_path(destination,(args.file,Path(__file__),DATA.parent/'SuPerPoV3.1.0.py'))
    except ValueError as error:
        parser.error(str(error))
    try:
        t=time.perf_counter(); df=pd.read_csv(args.file,sep='\t'); load_s=time.perf_counter()-t
        dates=None if args.all else (args.dates or ['1965-11-01','1965-12-23','1966-04-30'])
        t=time.perf_counter(); input_validation=validate_frame(df,RES,dates); validation_s=time.perf_counter()-t
    except (ValueError,OSError) as error:
        raise SystemExit(f'Invalid input: {error}') from error
    if args.all:
        dates=input_validation['dates']
    t=time.perf_counter(); tris=geometry(); geometry_s=time.perf_counter()-t
    t=time.perf_counter(); templates=[prepare_triangle_template(t) for t in tris]; template_s=time.perf_counter()-t
    values={date:group['data'].to_numpy() for date,group in df.groupby('# date',sort=False)}
    metadata={
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'environment':{'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),
                       'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'gudhi':gd.__version__},
        'file':str(args.file.resolve()),'file_size_bytes':args.file.stat().st_size,
        'resolution':RES,'triangles':[len(t) for t in tris],
        'data_load_s':load_s,'geometry_s':geometry_s,'template_preparation_s':template_s,'dates':len(dates),'repeats':args.repeats,
        'input_validation_s':validation_s,'input_validation':input_validation,'selected_dates':dates,
        'methods':selected_methods,'method_order':'rotates selected-method start by repetition index',
        'min_persistence':EPS,'homology_coeff_field':11,
        'diagram_endpoint_atol':1e-8,'score_atol':1e-10,
        'exactness':'numeric equality, not bit-pattern equality; signed zeros compare equal',
        'exit_requirement':'exact' if args.require_exact else 'tolerance',
        'score_matching':'exact death-value equality; first matching grid bar after decreasing-lifetime sort',
        'timing_includes':'both fresh simplex trees, vertex assignment, filtration, persistence, bar sorting, scores',
        'timing_excludes':'file load, input validation, geometry, template preparation, imports, process startup; no warm-up removal',
    }
    print(json.dumps(metadata),flush=True)
    results=[]; all_comparisons=[]
    for date in dates:
        z=values[date]
        timings={method:[] for method in selected_methods}; phases={method:[] for method in selected_methods}
        comparisons_by_repeat=[]
        first_reference=None
        for repeat in range(args.repeats):
            start=repeat%len(selected_methods)
            methods=selected_methods[start:]+selected_methods[:start]
            outputs={}
            for method in methods:
                phase={}
                t=time.perf_counter(); outputs[method]=calculate(tris,z,method,templates,phase); timings[method].append(time.perf_counter()-t)
                phases[method].append(phase)
            if first_reference is None:
                first_reference=outputs['extended']
            comparisons={m:compare(outputs['extended'],outputs[m]) for m in selected_methods if m!='extended'}
            reference_check=compare(first_reference,outputs['extended'])
            checks=[reference_check,*comparisons.values()]
            all_comparisons.extend(checks)
            comparisons_by_repeat.append({'repeat':repeat+1,'method_order':methods,
                                          'comparisons':comparisons,'reference_vs_first':reference_check,
                                          'verification':verification_status(checks,args.require_exact)})
        date_checks=[c for repetition in comparisons_by_repeat
                     for c in [repetition['reference_vs_first'],*repetition['comparisons'].values()]]
        report={'date':date,'comparisons_by_repeat':comparisons_by_repeat,
                'verification':verification_status(date_checks,args.require_exact),
                'seconds':timings,'median_seconds':{m:float(np.median(ts)) for m,ts in timings.items()},
                'phase_seconds':phases,'median_phase_seconds':{m:{key:float(np.median([p[key] for p in ps])) for key in ps[0]} for m,ps in phases.items()}}
        results.append(report)
        if args.all:
            print(json.dumps({'date':date,'verification':report['verification'],'median_seconds':report['median_seconds']}),flush=True)
        else:
            print(json.dumps(report),flush=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    verification=verification_status(all_comparisons,args.require_exact)
    destination.write_text(json.dumps({'metadata':metadata,'results':results,'verification':verification},indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'results_path':str(destination),'verification':verification}),flush=True)
    if not verification['passed']:
        raise SystemExit(f"Persistence comparison failed ({verification['requirement']} criterion); see {destination}")

if __name__=='__main__':
    main()
