# Sourced by setup_sam3_env.sh and run_sam3_reference.sh; the only shell reader of
# the settings of the fast SAM 3 reference: the [outlines] section of
# configs/models.toml, or of the file named by SAM3_REFERENCE_SETTINGS. Expects the
# workshop directory in $workshop.
#
#   sam3_setting <key>     prints the value, with a leading ~ expanded

sam3_settings_file="${SAM3_REFERENCE_SETTINGS:-$workshop/configs/models.toml}"

sam3_setting() {
  /usr/bin/python3 -c '
import os, sys, tomllib
with open(sys.argv[1], "rb") as handle:
    print(os.path.expanduser(tomllib.load(handle)["outlines"][sys.argv[2]]))' \
    "$sam3_settings_file" "$1"
}

# The section states where SAM 3 runs; the reference runs on the integrated graphics only.
sam3_check_device() {
  local device
  device="$(sam3_setting device)"
  if [[ "$device" != igpu ]]; then
    echo "SAM 3 runs on the integrated graphics only: outlines.device is '$device' in $sam3_settings_file, not 'igpu'" >&2
    return 1
  fi
}
