import json, os, glob
for path in sorted(glob.glob('notebooks/*.ipynb')):
    print(f'===== {os.path.basename(path)} =====')
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells'], 1):
        ct = cell.get('cell_type')
        src = ''.join(cell.get('source', []))
        print(f'--- {ct} cell {i} ---')
        print(src[:2600].strip())
        print('')
    print('\n')
