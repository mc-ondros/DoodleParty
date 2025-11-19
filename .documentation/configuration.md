# Configuration Reference

## AdminConfig.json

The core game configuration is stored in `AdminConfig.json` at the repository root. This file is shared between the frontend server and admin panel.

| Key | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `Timer` | Number | Game round duration in seconds. | `300` |
| `TimerPreset` | String | Preset label for UI. | `"300"` |
| `Game Mode` | String | Current game mode (e.g., "Classic", "Speed"). | `"Classic"` |
| `Max Players` | Number | Maximum concurrent players allowed. | `8` |
| `Ink Limit` | String | Ink usage limit (e.g., "Unlimited", "Medium"). | `"Unlimited"` |
| `Teams` | String | Team mode setting ("disabled" or enabled). | `"disabled"` |
| `Visibility` | String | Lobby visibility ("Public", "Private"). | `"Public"` |
| `Password` | String | Lobby password (empty for none). | `""` |
| `Custom Prompt` | String | Custom drawing prompt. | `""` |
| `Content Mode` | String | Content moderation level ("SFW", "NSFW"). | `"SFW"` |
| `Session` | String | Session state ("Open", "Closed"). | `"Open"` |

## Environment Variables

The Game Server (`frontend/server`) supports the following environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PORT` | Server listening port. | `3000` |
| `HOST` | Server bind address. | `0.0.0.0` |
| `ADMIN_USER` | Username for Admin UI Basic Auth. | `admin` |
| `ADMIN_PASSWORD` | Password for Admin UI Basic Auth. | See `frontend/server/express-server.cjs` for default |
| `DEMO_MODE` | Enable demo data generation (simulated drawings). | `0` (Disabled) |
