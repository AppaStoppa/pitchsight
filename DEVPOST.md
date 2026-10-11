# Devpost write-up (paste-ready: only track, side challenges and the three links need filling)

## Title
PitchSight

## Tagline
An AI analyst that watches a game clip in any of eight sports and tells every player what they did right, what they did wrong, and where they should have been.

## Track / side challenges
Main track: [pick]  ·  Side challenges (max 3): [pick]

## Team
Zain Kashif (solo)

## The problem and who it serves
Professional match-analysis tools are built and priced for pro clubs. School, youth and amateur teams rarely have
access to anything like them: coaches film games on a phone and then guess at what went wrong.
PitchSight gives them a first-pass tactical report from a single clip: shape, per-player ratings, positioning
errors and the run a player should have made, in the language of their own sport.

## What it does
- Pick a game from a searchable library, or upload your own clip. The sport is recognised automatically
  (soccer, basketball, American football, ice hockey, tennis, volleyball, baseball, cricket), or you choose it.
- Tracks every player and the ball, keeps the right number of players for the sport (a tennis clip is two players,
  not the crowd), and splits the sides by kit colour or by side of the net.
- Reads each side's shape on that sport's own surface: formation lines on a pitch, depth bands on a court, infield
  vs outfield in baseball.
- Gives each player a score out of 10 with what they did right, what they did wrong, nitpicks, an ideal position and
  a suggested run, plus key moments for the clip. The report is deliberately blunt and says when a player was barely seen.
- Shows it all on the video (spotlight or boxes, ideal-position ring, run arrow) and on an interactive tactical board.
- Confidence is shown, not hidden: team-split confidence per player (with a one-click "Wrong team?" fix) and how
  often the ball was actually seen.
- Visitors can save their own clips privately in their browser; nothing personal is stored on the server.

## How we built it
I started with a soccer-only prototype: YOLOv8 + ByteTrack for the players, KMeans on shirt colour for the two
teams, and a vision model for the coaching text. Real clips broke it in useful ways. A drone's-eye view turns players
into a few pixels that a person detector cannot see, so I wrote a camera-stabilised motion tracker. The "ball" it kept
finding was a traffic cone. A tennis clip counted three players. So I made it honest (it reports how often the ball
was really seen, and how sure the team split is), then generalised it to eight sports with one profile and one
vocabulary per sport, plus automatic sport detection.

In the last hours I tried a hand-held sideline football clip and the tracker "found" the bench, the mascot and the
cheerleaders, and missed the actual players. That led to the biggest rebuild: my own distance-based tracker (the
built-in one drops small, bunched-up players), a pan-and-zoom camera model, a playing-area filter that learns the
surface from where the moving players stand, and gap-filling so the numbers over the players stop blinking.

Pipeline: CLIP recognises the sport; YOLOv8 + ByteTrack track players and ball (with a motion tracker for steady
top-down drone footage that a photo-trained model cannot see); KMeans on torso and shorts colour splits the teams;
clustering on playing-surface depth gives the shape; per-player movement stats are turned into ratings and
sport-specific coaching text, and Claude vision writes the full report when an API key is present.
FastAPI back end, vanilla JS/Canvas front end. Results are cached and precomputed clips are served instantly, so the
demo works without a GPU or an API key.

## Challenges
- **Tracking from above.** A drone's-eye view turns people into a few pixels; YOLO sees nothing. We built a
  camera-stabilised motion tracker (median background, phase-correlation drift removal) and a detector probe that
  chooses between the models per clip.
- **Finding the ball honestly.** Early versions "saw" the ball 97% of the time: it was a traffic cone. We added
  static-object suppression and a path search, and now only make ball-based remarks when the ball was really seen.
- **Counting players.** Tennis kept returning three players and fragments of one player under several IDs. We stitch
  broken tracks and choose the player count per sport from how sharply the evidence drops off.
- **Telling players from the crowd.** On a sideline football clip the first version "tracked" the bench, the mascot
  and the cheerleaders and missed the actual players. We now learn the playing surface from where the moving players
  stand, track the camera's pan and zoom, and ignore everyone outside the playing area, and the site says how many
  people it ignored.
- **Fair ratings.** A far-court player looks like they barely move; we normalise by body size and compare players
  only to others in the same clip.
- **Sounding like the sport.** A tennis report must not talk about a back four, so every sport has its own
  vocabulary and an automated test fails if soccer words leak into another sport.
- **Telling teams apart** when kits are similar: confidence per player and a manual correction.

## What we learned / what's next
Next: pitch-keypoint homography for true court coordinates, jersey-number OCR so ratings follow named players,
more footage per sport to tune the profiles, and season-long player trends.

## Built with
Python, FastAPI, YOLOv8 (Ultralytics), ByteTrack, OpenCV, CLIP (Transformers), scikit-learn, NumPy, Anthropic Claude API, JavaScript, HTML Canvas

## Disclosure (made before Hacklanta)
Nothing in this project existed before the event. Pretrained YOLOv8 weights, CLIP and open-source libraries are used as-is.
Built with AI coding assistance (Claude).

## Judge instructions
No credentials needed. Open the demo link and click any clip in the Game library, or upload a short clip of any of the
eight sports. Direct links open a result immediately, for example `/?game=volleyball-clip-1` or `/?game=tennis-1-on-1-clip`.
Repo README has run instructions.

## Links
Demo: [url]  ·  Repo (PUBLIC): [url]  ·  Video (PUBLIC, 2-3 min): [url]
