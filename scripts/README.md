# How to use
At the start of each term, do the following
1. Archive the old leaderboard Markdown file manually and add a link to it in the [README_template.md](https://github.com/utiasDSL/lsy_drone_racing/blob/competition/scripts/README_template.md)
1. Remove all previous teams from the [leaderboard.toml](https://github.com/utiasDSL/lsy_drone_racing/blob/competition/scripts/leaderboard.toml)
1. Create new [`DISPATCH_TOKEN`](https://github.com/settings/personal-access-tokens) valid for the full term period
1. Add teams that have registered for the competition to the [leaderboard.toml](https://github.com/utiasDSL/lsy_drone_racing/blob/competition/scripts/leaderboard.toml) to whitelist them. Only add the `name` and `github` field, the rest will be added by the scripts.

# Workflow
After a team is whitelisted, the workflow of the competition is as follows
1. Team pushes to the main branch in their fork
1. A `repository_dispatch` is sent to the central repo
1. The central repo clones the code of the team, runs the evaluation, and updates the leaderboard
