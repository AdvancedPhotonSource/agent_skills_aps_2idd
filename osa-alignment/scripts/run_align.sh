osa_root=/home/beams/USER2IDD/software/workspaces/2026-3/.pi/skills/osa-alignment/scripts
osa_run=$(mktemp -d "$osa_root/workspace/run.XXXXXX")

uv run --project "$osa_root" --locked \
  python "$osa_root/align_osa.py" \
  --config "$osa_root/alignment.json" \
  --adapter aps2idd_adapter:create_adapter \
  --adapter-config "$osa_root/aps2idd.adapter.json" \
  --log "$osa_run/trajectory.jsonl" \
  --output "$osa_run/result.json" \
  --webui --webui-port 8765
