# Deploy Tensors to Junkpile

Build, deploy, and restart tensors on junkpile with verification.

Run the deploy script:

```bash
./scripts/deploy.sh
```

## What it does

1. **Build UI** - Runs `npm run build` in `tensors/server/ui/`
2. **Sync code** - Rsyncs Python code to `/opt/tensors/app/`
3. **Fix permissions** - Sets ownership to `tensors:tensors`
4. **Restart tensors** - Runs `sudo systemctl restart tensors`
5. **Verify tensors** - Checks service is running

## Service Structure

| Item | Value |
|------|-------|
| User/Group | `tensors:tensors` |
| Install path | `/opt/tensors/app` |
| Venv | `/opt/tensors/venv` |
| Service | `tensors.service` |
| Port | 8081 |

## Access

- **Local**: http://junkpile:8081
