import json
from pathlib import Path
import re

root = Path(__file__).parent
notebooks = [
    root / 'EQE' / 'Solar_Cell_Analysis.ipynb',
    root / 'EQE' / 'EQE_Analysis.ipynb',
    root / 'UV-Vis' / 'Corrected_UV_Vis_analysis.ipynb',
    root / 'UV-Vis' / 'UV_Vis_analysis.ipynb',
    root / 'SCAPS Simulations' / 'SCAPS_Simulations_iv_files.ipynb',
    root / 'SCAPS Simulations' / 'SCAPS_Simulations_iv_files copy.ipynb',
    root / 'SCAPS Simulations' / 'SCAPS_Simulations_v1_BACKUP.ipynb',
    root / 'SCAPS Simulations' / 'SCAPS_Simulations_v1_for_1_plot_scaps.ipynb',
]

helper_import = (
    'import sys\n'
    'from pathlib import Path\n'
    'current_path = Path.cwd().resolve()\n'
    'for parent in [current_path] + list(current_path.parents):\n'
    '    if (parent / \'project_paths.py\').exists():\n'
    '        sys.path.insert(0, str(parent))\n'
    '        break\n'
    '\n'
    'from project_paths import data_path\n'
)

absolute_base = 'C:/Users/jp_ol/OneDrive/Ambiente de Trabalho/TESE/Data_Analysis/data/'
absre = re.compile(r'(?P<quote>["\"])' + re.escape(absolute_base) + r'(?P<path>[^"\']*)(?P=quote)')


def convert_path(path):
    parts = [p for p in path.replace('\\', '/').split('/') if p]
    if not parts:
        return None
    return 'data_path(' + ', '.join(f'"{p}"' for p in parts) + ')'

for nb_path in notebooks:
    if not nb_path.exists():
        print('MISSING', nb_path)
        continue
    data = json.loads(nb_path.read_text(encoding='utf-8'))
    modified = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new_src = src
        if 'from project_paths import data_path' not in new_src and 'from pathlib import Path' in new_src:
            if 'import sys' not in new_src:
                new_src = new_src.replace('from pathlib import Path\n', 'from pathlib import Path\nimport sys\n')
            if 'from project_paths import data_path' not in new_src:
                new_src = new_src.replace('from pathlib import Path\nimport sys\n', 'from pathlib import Path\nimport sys\nfrom project_paths import data_path\n')
        new_src = new_src.replace('current_path = Path.cwd().resolve()\n    data_path = current_path.parent', helper_import)
        new_src = new_src.replace(
            'current_path = Path.cwd().resolve()\nfor parent in [current_path] + list(current_path.parents):\n    if (parent / \'project_paths.py\').exists():\n        sys.path.insert(0, str(parent))\n        break\n\nfrom project_paths import data_path\n',
            helper_import,
        )
        def repl(m):
            conv = convert_path(m.group('path'))
            return conv if conv is not None else m.group(0)
        new_src = absre.sub(repl, new_src)
        if 'eqe_data / "data" / "2nd Round" / "Corrected" / "Mo_200Sb_25Se_Results Table.xlsx"' in new_src:
            new_src = new_src.replace(
                'eqe_data / "data" / "2nd Round" / "Corrected" / "Mo_200Sb_25Se_Results Table.xlsx"',
                'data_path("2nd Round", "Corrected", "Mo_200Sb_25Se_Results Table.xlsx")',
            )
        if new_src != src:
            modified = True
            cell['source'] = [line + '\n' for line in new_src.splitlines()]
    if modified:
        nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        print('UPDATED', nb_path.name)
    else:
        print('NO CHANGE', nb_path.name)
