import os
import re

extensions = ['.py', '.c', '.sv']

root = r'c:\\intelFPGA_lite\\DSD\\Group_project'

for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        _, ext = os.path.splitext(fname)
        if ext.lower() in extensions:
            file_path = os.path.join(dirpath, fname)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            new_lines = []
            for line in content.splitlines():
                stripped = line.lstrip()
                if stripped.startswith('
                    continue
                if '
                    line = line.split('
                if '
                    if line.lstrip().startswith('
                        new_lines.append(line)
                        continue
                    line = line.split('
                new_lines.append(line.rstrip())
            new_content = "\n".join(new_lines) + "\n"
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Stripped comments from {file_path}")
