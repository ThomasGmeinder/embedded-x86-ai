# SO-ARM101 Calibration

Calibrate an arm in one step.

## First time only

```bash
pip install pyserial
```

## Calibrate

1. Plug the arm into USB.
2. Run:

   ```bash
   sudo python3 calibrate.py
   ```

3. When prompted, move the arm to its **home pose** (all joints centered), hold
   it, and press **Enter**.

Done — it detects the arm, writes the calibration to the servos, verifies it,
and saves a `calibrated_<serial>.json` file next to the script.

---

<details>
<summary>Advanced / troubleshooting</summary>

- **Permission error?** The command uses `sudo` because it needs USB bus access.
  (Alternative: `sudo usermod -aG dialout $USER`, re-login, then drop `sudo`.)
- **Inspect an arm without changing it:**
  ```bash
  sudo python3 read_calibration.py --port <by-id path>
  ```
- **Manual control** (custom reference/output, or dry-run without writing):
  ```bash
  # dry run (no writes)
  sudo python3 calibrate_from_reference.py --port <by-id path> \
      --ref golden_reference.json --out arm.json
  # commit to the servos
  sudo python3 calibrate_from_reference.py --port <by-id path> \
      --ref golden_reference.json --out arm.json --commit
  ```
- `golden_reference.json` is the known-good reference. Joint ranges come from it;
  each arm's homing offset is measured live from the home pose.

</details>
