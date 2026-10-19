#!/bin/bash
set -e

# Loop through directories 1 to 5
for dir in 1 2 3 4 5; do
    if [ ! -d "$dir" ]; then
        echo "[-] Directory $dir not found. Skipping..."
        continue
    fi

    echo "=================================================="
    echo "[+] Processing Case Study Directory: $dir"
    echo "=================================================="
    cd "$dir"

    # 1. Standardize Target file name (handles Target.ent in folder 2)
    if [ -f "Target.ent" ] && [ ! -f "Target.pdb" ]; then
        cp Target.ent Target.pdb
    fi

    # 2. Clean protein receptors and convert to PDBQT
    for prot in Target Off; do
        if [ -f "${prot}.pdb" ]; then
            python3 -c "
with open('${prot}.pdb', 'r') as f_in, open('${prot}_clean.pdb', 'w') as f_out:
    for line in f_in:
        if line.startswith('ATOM'):
            f_out.write(line)
"
            obabel ${prot}_clean.pdb -O ${prot}.pdbqt -p 7.4 2>/dev/null
            
            # Remove Open Babel root/branch tags for rigid receptors
            grep -v -E "^(ROOT|ENDROOT|BRANCH|ENDBRANCH|TORSDOF)" ${prot}.pdbqt > ${prot}_rigid.pdbqt
            mv ${prot}_rigid.pdbqt ${prot}.pdbqt
        fi
    done

    # Function to calculate grid box parameters
    get_grid() {
        local pdb_file=$1
        python3 -c "
import numpy as np
coords = []
with open('$pdb_file') as f:
    for line in f:
        if line.startswith('ATOM'):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
coords = np.array(coords)
center = coords.mean(axis=0)
size = coords.max(axis=0) - coords.min(axis=0) + 10.0
print(f'{center[0]:.3f},{center[1]:.3f},{center[2]:.3f},{min(size[0], 126):.3f},{min(size[1], 126):.3f},{min(size[2], 126):.3f}')
"
    }

    # Helper function to write Vina config and execute
    run_vina_docking() {
        local ligand=$1
        local receptor=$2
        local out_name=$3
        local grid_params=$4

        IFS=',' read -r cx cy cz sx sy sz <<< "$grid_params"

        cat << EOF > ${out_name}.cfg
receptor = ${receptor}.pdbqt
ligand = ${ligand}.pdbqt

center_x = $cx
center_y = $cy
center_z = $cz

size_x = $sx
size_y = $sy
size_z = $sz

energy_range = 3
exhaustiveness = 16
num_modes = 9
out = ${out_name}_poses.pdbqt
EOF

        echo "  [->] Docking $ligand against $receptor -> ${out_name}_poses.pdbqt"
        vina --config ${out_name}.cfg > ${out_name}.log 2>&1
    }

    # Calculate grids
    target_grid=$(get_grid "Target_clean.pdb")
    off_grid=$(get_grid "Off_clean.pdb")

    # Run 1: main_phyto + Target
    run_vina_docking "main_phyto" "Target" "dock_main_Target" "$target_grid"

    # Run 2: main_phyto + Off
    run_vina_docking "main_phyto" "Off" "dock_main_Off" "$off_grid"

    # Run 3: similar_phyto + Target
    run_vina_docking "similar_phyto" "Target" "dock_similar_Target" "$target_grid"

    cd ..
    echo "[+] Completed Directory: $dir"
    echo ""
done

echo "=================================================="
echo "[+] All batch docking runs completed successfully!"
echo "=================================================="
