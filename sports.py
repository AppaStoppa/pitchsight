"""Sport profiles. Everything that differs between sports lives here as plain data, so the
pipeline, the prompt and the API can stay sport-agnostic.

Fields
  label        display name
  unit         what a "team" is called in the UI (Team, or Side for net sports)
  surface      which playing surface the tactics board draws
  max_players  upper bound on tracked people we keep (players, not coaches or spectators)
  count_range  (fewest, most) players that can plausibly be in a clip; the real number is chosen inside it
  balanced     True when both sides are always the same size (basketball, hockey)
  depth_axis   frame axis the "lines" run along: "x" for a side-on view, "y" for tennis broadcast
  team_split   "colour" (KMeans on kit colour) or "side" (which half of the court/net)
  formation    which estimator describes the shape: soccer | lines | ring | tennis
  lines        number of depth bands for the "lines" estimator
  ring_words   optional (inner, outer) words for the "ring" estimator, default ("close", "deep")
  static_ok    True when real players may stand still for most of a clip (fielders in baseball and cricket), so
               "never moves" is not used as evidence that someone is a bystander
  role         default role label when no AI report is available
  context      short description handed to the vision model
  focus        what the model should judge
"""

SPORTS = {
    "soccer": dict(
        label="Soccer", count_range=(4, 22), balanced=False, unit="Team", surface="pitch", max_players=22, depth_axis="x",
        team_split="colour", formation="soccer", lines=4, role="Outfield player",
        context="Association football, 11 v 11 on a large pitch. Describe formations as lines of "
                "players from defence to attack (for example 4-3-3).",
        focus="positioning and spacing, line height, pressing angles, runs in behind, "
              "body shape when receiving, tracking back",
    ),
    "basketball": dict(
        label="Basketball", count_range=(4, 10), balanced=True, unit="Team", surface="court", max_players=10, depth_axis="x",
        team_split="colour", formation="lines", lines=2, role="Player",
        context="Basketball, 5 v 5 on a court with a basket at each end. Describe the shape as how many "
                "players sit in each band from one basket to the other (for example 3-2) and name the "
                "actions: spacing, cuts, screens, help defence, transition.",
        focus="spacing and floor balance, off-ball cuts, screening actions, help-side defence, "
              "transition defence, closing out shooters",
    ),
    "football": dict(
        label="American football", count_range=(8, 22), balanced=False, unit="Team", surface="gridiron", max_players=22, depth_axis="x",
        team_split="colour", formation="lines", lines=3, role="Player",
        context="American football, 11 v 11 with offence and defence lined up either side of the line of "
                "scrimmage. Describe each unit's shape by depth (line, second level, secondary).",
        focus="alignment and depth, gap and run fits, coverage leverage, route spacing, "
              "pursuit angles, blocking angles",
    ),
    "hockey": dict(
        label="Ice hockey", count_range=(4, 12), balanced=True, unit="Team", surface="rink", max_players=12, depth_axis="x",
        team_split="colour", formation="lines", lines=3, role="Skater",
        context="Ice hockey, 6 v 6 including goalies, on a rink. Describe shape by depth (defence, middle, "
                "forwards) and judge structure such as forecheck, neutral-zone trap, breakouts, gap control.",
        focus="gap control, support for the puck carrier, forecheck and backcheck structure, "
              "passing lanes, net-front presence",
    ),
    "tennis": dict(
        label="Tennis", count_range=(2, 4), balanced=False, unit="Side", surface="tennis", max_players=4, depth_axis="y",
        team_split="side", formation="tennis", lines=2, role="Player",
        context="Tennis, singles or doubles. A 'side' is one half of the court; players are assigned to a "
                "side by where they stand relative to the net. Describe the set-up (for example one up, "
                "one back, or both back).",
        focus="court position, recovery to the centre after each shot, split-step timing, "
              "net approach, doubles spacing and poaching",
    ),
    "volleyball": dict(
        label="Volleyball", count_range=(4, 12), balanced=False, unit="Side", surface="volleyball", max_players=12, depth_axis="x",
        team_split="side", formation="lines", lines=2, role="Player",
        context="Volleyball, 6 v 6 either side of a net. Describe the shape as front row / back row "
                "counts (for example 3-3).",
        focus="serve receive shape, block positioning, coverage of hitters, rotations, "
              "transition from defence to attack",
    ),
    "baseball": dict(
        label="Baseball", count_range=(4, 13), balanced=False, unit="Team", surface="diamond", max_players=13, depth_axis="y",
        team_split="colour", formation="ring", lines=2, role="Fielder", ring_words=("infield", "outfield"),
        static_ok=True,
        context="Baseball on a diamond. People fall into two groups: the fielding team (pitcher, catcher, "
                "infielders, outfielders) and the batter, runners, coaches and umpires. Describe the fielding "
                "set-up as how many fielders play the infield against the outfield, and judge shifts, "
                "positioning for the batter, cut-off alignments and base coverage.",
        focus="defensive shifts and depth for the batter, cut-off and relay alignment, base coverage, "
              "backing up throws, outfield spacing, baserunner leads and reads",
    ),
    "cricket": dict(
        label="Cricket", count_range=(4, 14), balanced=False, unit="Side", surface="oval", max_players=14, depth_axis="x",
        team_split="colour", formation="ring", lines=2, role="Fielder", static_ok=True,
        context="Cricket on an oval with a central pitch. People fall into two groups: the fielding side, "
                "and the batters and umpires. Describe the fielding set-up as how many fielders are close "
                "in (catching, saving singles) against deep (boundary cover).",
        focus="field placement for the bowler's line, catching positions, backing up, "
              "boundary cover, gaps in the ring",
    ),
}

DEFAULT_SPORT = "soccer"


def get(sport):
    """Return the profile for a sport key, or raise ValueError listing the valid keys."""
    key = (sport or DEFAULT_SPORT).lower()
    if key not in SPORTS:
        raise ValueError(f"Unknown sport '{sport}'. Choose one of: {', '.join(SPORTS)}")
    return key, SPORTS[key]
