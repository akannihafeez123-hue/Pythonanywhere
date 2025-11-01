```markdown
# Hidden_bot — Clever Cloud deployment ready

This branch includes:
- Updated Hidden_bot compatible with python-telegram-bot v20+ (synchronous calls via Request).
- WSGI entrypoint (wsgi.py) so background tasks start under gunicorn.
- Procfile and clevercloud.json for deployment.
- requirements.txt and .env.example.

How to create the PR (local steps)
1. Ensure you're on your repo root and you have the branch created:
   git checkout -b clevercloud-deploy

2. Add files and commit:
   git add Hidden_bot wsgi.py Procfile requirements.txt clevercloud.json .env.example README.md
   git commit -m "Adapt Hidden_bot to python-telegram-bot v20+, add WSGI + Clever Cloud config"

3. Push branch:
   git push origin clevercloud-deploy

4. Create a Pull Request:
   - Open GitHub > your repository > "Compare & pull request" for the clevercloud-deploy branch.
   - PR title suggestion: "chore: adapt bot to ptb v20 and add Clever Cloud deployment files"
   - PR description: mention changes (Hidden_bot update, wsgi entrypoint, Procfile, requirements, clevercloud.json).

Or create PR from command line (if using hub or gh CLI):
   gh pr create --base main --head clevercloud-deploy --title "chore: adapt bot to ptb v20 and add Clever Cloud deployment files" --body "See changes: updated Hidden_bot, wsgi entrypoint, Procfile, requirements, clevercloud.json."

Clever Cloud deployment steps (web UI)
1. Create an account / log in to Clever Cloud: https://www.clever-cloud.com
2. Create a new application:
   - Choose "Python" as the runtime.
   - When prompted, you can either link your GitHub repo or choose "Push to Clever Cloud" and add a Git remote.

3. If linking GitHub:
   - Choose your repository and select the branch clevercloud-deploy.
   - Clever Cloud will detect requirements.txt and run pip install.

4. If using git push:
   - Add the Clever Cloud remote (Clever Cloud provides it when creating app):
     git remote add clever <provided-remote-url>
   - Push your branch:
     git push clever clevercloud-deploy:master

5. In the Clever Cloud dashboard, set environment variables:
   - TELEGRAM_TOKEN (your bot token)
   - TELEGRAM_CHAT_ID (chat id or @channelusername)
   - STORJ_API (optional)
   - VASTAI_API (optional)
   - You can also set KEEP_ALIVE_INTERVAL, EARNINGS_CHECK_INTERVAL, DAILY_REPORT_INTERVAL if you want non-default values.

6. Deployment will run automatically. Gunicorn will use wsgi:application as the entrypoint because of the Procfile.

Clever Cloud with GitHub integration:
- If you linked GitHub, you can enable automatic deployments on push to clevercloud-deploy.

Testing & verifying
- After deployment, verify logs in Clever Cloud (console > app > logs).
- Test endpoints:
  - GET /ping
  - POST /microservices/sale with JSON {"service":"service_1","amount":1.23}
- Check Telegram for notifications.

Security note
- Keep TELEGRAM_TOKEN and API keys out of git. Use Clever Cloud env vars for secrets.

If you’d like, I can:
- Open the PR for you (I created the branch already). If you want me to open it, I will push the files onto the branch and create the PR.
- Or I can provide the exact git commands I will run to push the files and open the PR (then you can run them locally or let me run them after you confirm).

```