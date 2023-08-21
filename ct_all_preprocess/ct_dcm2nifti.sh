read -p "Enter the main directory path [contains all patient files]: " directory
cd "$directory" || { echo "Erro: Directory not found"; exit 1; }
for subfolder in */; do
  cd "$subfolder" || { echo "Error: Subfolder '$subfolder' not found"; continue; }
  echo "${subfolder%?}"
  for sub1folder in */; do
    cd "$sub1folder" || { echo "Error: Subfolder '$sub1folder' not found"; continue; }
    dcm2niix -o ../../ -z y -f "%i_${sub1folder%?}" ./
    rm -f ../../*.json
    cd ..
  done
  cd ..
  for file in *"${subfolder%?}"*.nii.gz; do
    mv "$file" "${subfolder%?}${file#_}"
  done
done


