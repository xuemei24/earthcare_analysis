varName="sulfate"
input_file1="_12.nc"

input_dir="/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/july/"
input_file0="_mmr_july_2025"
input_file=${input_dir}${varName}${input_file0}${input_file1}
dim_name="forecast_reference_time"   # Replace with your actual dimension name (e.g., time, step, etc.)

for i in $(seq 0 62); do
  output_file=${input_dir}"temp/"${varName}${input_file0}"_slice_${i}"${input_file1}
  echo "Creating $output_file with slice $i"
  ncks -d ${dim_name},${i},${i} "$input_file" "$output_file"
done

