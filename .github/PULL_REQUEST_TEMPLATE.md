---
name: Pull Request
about: Contribute a change to XMem
---

## Summary
<!-- What does this PR do? (1-3 sentences) -->


## Motivation / Problem
<!-- Why is this change needed? Link to the relevant issue. -->

Closes #<!-- issue number -->

## Changes
<!-- Bullet-point list of what you changed -->
- 

## Testing
<!-- How did you verify this works? -->
- [ ] Unit tests added / updated (`pytest tests/unit`)
- [ ] Integration tests pass (`pytest tests/integration`)
- [ ] Tested manually — steps below:

```
# command to reproduce
```

## Screenshots / recordings (if UI change)
<!-- Drag & drop a screenshot or screen recording -->

## Checklist
- [ ] My PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat(scope): description`)
- [ ] I ran `ruff check .` and `black --check .` locally with no errors
- [ ] I updated `CHANGELOG.md` if this is a user-visible change
- [ ] I ran `uv lock` if I modified `pyproject.toml`
- [ ] Security-sensitive files modified? Pinged `@ishaanxgupta` or `@ved015`
