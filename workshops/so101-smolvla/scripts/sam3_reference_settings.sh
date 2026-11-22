# Sourced by setup_sam3_env.sh and run_sam3_reference.sh; the only shell reader of
# the settings of the fast SAM 3 reference (configs/sam3_reference.json, or the file
# named by SAM3_REFERENCE_SETTINGS). Expects the workshop directory in $workshop.
#
#   sam3_setting <key>     prints the value, with a leading ~ expanded

sam3_settings_file="${SAM3_REFERENCE_SETTINGS:-$workshop/configs/sam3_reference.json}"

sam3_setting() {
  /usr/bin/python3 -c \
    'import json, os, sys; print(os.path.expanduser(json.load(open(sys.argv[1]))[sys.argv[2]]))' \
    "$sam3_settings_file" "$1"
}
