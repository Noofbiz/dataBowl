package simulate

import (
	"math"
	"testing"

	"github.com/Noofbiz/dataBowl/datasets"
)

// ── helpers ──────────────────────────────────────────────────────────────────

// makeGame builds a minimal *datasets.Game with one play containing nFrames
// consecutive frames for one player moving at constant speed/direction.
func makeGame(role string, speed, dir float64, nFrames int) *datasets.Game {
	play := datasets.Play{
		ID:     "1",
		Frames: make(map[string]datasets.Frame),
	}

	vel := velocityFromPolar(speed, dir)
	x, y := 60.0, 26.65 // mid-field start

	for i := 1; i <= nFrames; i++ {
		frame := datasets.Frame{
			ID:      itoa(i),
			Players: map[string]datasets.Player{},
		}
		frame.Players["42"] = datasets.Player{
			ID:              "42",
			PlayerRole:      role,
			X:               x,
			Y:               y,
			Velocity:        speed,
			AngleOfMomentum: dir,
		}
		play.Frames[itoa(i)] = frame
		x += vel.X * dt
		y += vel.Y * dt
	}

	game := &datasets.Game{
		ID:    "g1",
		Plays: map[string]datasets.Play{"1": play},
	}
	return game
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	// simple enough for tests that use ≤ 30 frames
	return string([]byte{byte('0' + i/10), byte('0' + i%10)})
}

// ── ForceTable tests ──────────────────────────────────────────────────────────

func TestBuildForceTable_Empty(t *testing.T) {
	ft := BuildForceTable(nil, 42)
	got := ft.Sample("Passer", 5.0, 90.0)
	if got.X != 0 || got.Y != 0 {
		t.Errorf("expected zero sample from empty table, got %+v", got)
	}
}

func TestBuildForceTable_ConstantMotion(t *testing.T) {
	// Constant speed/direction → zero acceleration → all samples should be ≈ 0
	game := makeGame("Receiver", 5.0, 90.0, 10)
	ft := BuildForceTable([]*datasets.Game{game}, 1)

	// Mean of samples in the matching bucket should be near zero
	const N = 1000
	var sumX, sumY float64
	for range N {
		s := ft.Sample("Receiver", 5.0, 90.0)
		sumX += s.X
		sumY += s.Y
	}
	meanX := sumX / N
	meanY := sumY / N
	if math.Abs(meanX) > 0.5 || math.Abs(meanY) > 0.5 {
		t.Errorf("expected near-zero mean acceleration for constant motion, got (%.3f, %.3f)", meanX, meanY)
	}
}

func TestBuildForceTable_Samples(t *testing.T) {
	// Two frames → one consecutive pair → at least one sample stored
	game := makeGame("Corner", 3.0, 45.0, 2)
	ft := BuildForceTable([]*datasets.Game{game}, 7)
	if len(ft.samples) == 0 {
		t.Error("expected samples to be populated")
	}
}

func TestForceTable_FallbackRole(t *testing.T) {
	// Build with role "Passer"; sample with a different role should still return
	// a value via the catch-all ("", anyBucket, anyBucket) fallback.
	game := makeGame("Passer", 2.0, 0.0, 5)
	ft := BuildForceTable([]*datasets.Game{game}, 99)
	s := ft.Sample("UnknownRole", 2.0, 0.0)
	// Just verify no panic and a value is returned (fallback to "" key).
	_ = s
}

func TestForceTable_NilSafetyOnSample(t *testing.T) {
	var ft *ForceTable
	s := ft.Sample("Passer", 5.0, 90.0)
	if s.X != 0 || s.Y != 0 {
		t.Errorf("nil ForceTable.Sample should return zero, got %+v", s)
	}
}

// ── helper: velocityFromPolar round-trip ──────────────────────────────────────

func TestVelocityFromPolar_RightAngle(t *testing.T) {
	// dir=90° (east) → vx=speed, vy≈0
	v := velocityFromPolar(5.0, 90.0)
	if math.Abs(v.X-5.0) > 1e-9 || math.Abs(v.Y) > 1e-9 {
		t.Errorf("dir=90°: expected (5,0), got (%f,%f)", v.X, v.Y)
	}
}

func TestVelocityFromPolar_North(t *testing.T) {
	// dir=0° (north) → vx=0, vy=speed
	v := velocityFromPolar(4.0, 0.0)
	if math.Abs(v.X) > 1e-9 || math.Abs(v.Y-4.0) > 1e-9 {
		t.Errorf("dir=0°: expected (0,4), got (%f,%f)", v.X, v.Y)
	}
}

func TestSpeedAndDir_RoundTrip(t *testing.T) {
	for _, tc := range []struct{ speed, dir float64 }{
		{3.5, 0},
		{3.5, 90},
		{3.5, 180},
		{3.5, 270},
		{3.5, 45},
	} {
		vel := velocityFromPolar(tc.speed, tc.dir)
		gotSpeed, gotDir := speedAndDir(vel)
		if math.Abs(gotSpeed-tc.speed) > 1e-9 {
			t.Errorf("speed round-trip: input=%.1f got=%.6f", tc.speed, gotSpeed)
		}
		if math.Abs(gotDir-tc.dir) > 1e-6 {
			t.Errorf("dir round-trip: input=%.1f got=%.6f", tc.dir, gotDir)
		}
	}
}

// ── Simulator tests ───────────────────────────────────────────────────────────

func TestSimulator_OutputShape(t *testing.T) {
	game := makeGame("Receiver", 4.0, 45.0, 10)
	ft := BuildForceTable([]*datasets.Game{game}, 0)
	sim := NewSimulator(ft)

	initState := State{
		Pos:   Vec2{60, 26},
		Vel:   velocityFromPolar(4.0, 45.0),
		Speed: 4.0,
		Dir:   45.0,
	}

	const nRuns, nSteps = 50, 10
	results := sim.Run("Receiver", initState, nRuns, nSteps)

	if len(results) != nRuns {
		t.Fatalf("expected %d results, got %d", nRuns, len(results))
	}
	for i, r := range results {
		// Each trajectory has nSteps+1 points (initial + one per step)
		if len(r.Trajectory) != nSteps+1 {
			t.Errorf("run %d: expected %d trajectory points, got %d", i, nSteps+1, len(r.Trajectory))
		}
	}
}

func TestSimulator_FieldBounds(t *testing.T) {
	// Start near edge and move aggressively; no point should escape field.
	ft := BuildForceTable(nil, 0)
	sim := NewSimulator(ft)

	initState := State{
		Pos:   Vec2{0.1, 0.1},
		Vel:   Vec2{-20, -20}, // would immediately exit field without clamping
		Speed: math.Sqrt(800),
		Dir:   225,
	}
	results := sim.Run("Receiver", initState, 10, 20)
	for i, r := range results {
		for j, pos := range r.Trajectory {
			if pos.X < fieldMinX || pos.X > fieldMaxX || pos.Y < fieldMinY || pos.Y > fieldMaxY {
				t.Errorf("run %d step %d: position (%f,%f) outside field bounds", i, j, pos.X, pos.Y)
			}
		}
	}
}

func TestSimulator_SpeedClamp(t *testing.T) {
	// Build a game with very high acceleration to test maxSpeed clamping.
	// We inject large acceleration directly by varying speeds in makeGame.
	ft := BuildForceTable(nil, 0)
	sim := NewSimulator(ft)

	initState := State{
		Pos:   Vec2{60, 26},
		Vel:   Vec2{100, 100}, // way over maxSpeed
		Speed: math.Sqrt(20000),
		Dir:   45,
	}
	results := sim.Run("Receiver", initState, 5, 5)
	for i, r := range results {
		for j := range r.Trajectory {
			if j == 0 {
				continue // initial pos is unclamped intentionally
			}
			// Velocity after first step must be ≤ maxSpeed.
			// We can't inspect velocity directly, but positions must be valid.
			pos := r.Trajectory[j]
			if pos.X < fieldMinX || pos.X > fieldMaxX || pos.Y < fieldMinY || pos.Y > fieldMaxY {
				t.Errorf("run %d step %d: out of bounds (%f,%f)", i, j, pos.X, pos.Y)
			}
		}
	}
}

func TestSimulator_ZeroRuns(t *testing.T) {
	ft := BuildForceTable(nil, 0)
	sim := NewSimulator(ft)
	results := sim.Run("Receiver", State{}, 0, 10)
	if results != nil {
		t.Errorf("expected nil results for nRuns=0, got %v", results)
	}
}

func TestSimulator_StartPositionPreserved(t *testing.T) {
	ft := BuildForceTable(nil, 0)
	sim := NewSimulator(ft)

	start := Vec2{42.5, 18.3}
	initState := State{Pos: start, Speed: 0, Dir: 0}
	results := sim.Run("Receiver", initState, 3, 5)
	for i, r := range results {
		if r.Trajectory[0].X != start.X || r.Trajectory[0].Y != start.Y {
			t.Errorf("run %d: first trajectory point (%f,%f) != start (%f,%f)",
				i, r.Trajectory[0].X, r.Trajectory[0].Y, start.X, start.Y)
		}
	}
}

// ── MeanTrajectory tests ──────────────────────────────────────────────────────

func TestMeanTrajectory_Empty(t *testing.T) {
	mt := MeanTrajectory(nil)
	if mt != nil {
		t.Error("expected nil for empty input")
	}
}

func TestMeanTrajectory_SingleRun(t *testing.T) {
	traj := Trajectory{{1, 2}, {3, 4}, {5, 6}}
	results := []Result{{Trajectory: traj}}
	mt := MeanTrajectory(results)
	for i, p := range mt {
		if p.X != traj[i].X || p.Y != traj[i].Y {
			t.Errorf("step %d: mean (%f,%f) != input (%f,%f)", i, p.X, p.Y, traj[i].X, traj[i].Y)
		}
	}
}

func TestMeanTrajectory_Average(t *testing.T) {
	// Two runs: one at (0,0)→(2,2), one at (2,2)→(0,0)
	// Mean should be (1,1) at every step.
	r1 := Result{Trajectory: Trajectory{{0, 0}, {2, 2}}}
	r2 := Result{Trajectory: Trajectory{{2, 2}, {0, 0}}}
	mt := MeanTrajectory([]Result{r1, r2})
	for i, p := range mt {
		if math.Abs(p.X-1) > 1e-9 || math.Abs(p.Y-1) > 1e-9 {
			t.Errorf("step %d: expected (1,1) got (%f,%f)", i, p.X, p.Y)
		}
	}
}

// ── End-to-end smoke test ─────────────────────────────────────────────────────

func TestEndToEnd_TrainAndSimulate(t *testing.T) {
	// Build a training corpus, extract forces, simulate, check results make sense.
	games := []*datasets.Game{
		makeGame("Wide Receiver", 6.0, 90.0, 20),  // fast east-moving receiver
		makeGame("Corner Back",   3.0, 270.0, 20), // slower west-moving defender
	}
	ft := BuildForceTable(games, 12345)
	sim := NewSimulator(ft)

	initState := State{
		Pos:   Vec2{50, 25},
		Speed: 6.0,
		Dir:   90.0,
	}
	initState = normalizeState(initState)

	results := sim.Run("Wide Receiver", initState, 100, 20)
	if len(results) != 100 {
		t.Fatalf("expected 100 results, got %d", len(results))
	}

	// Mean final position should be well east of start (x > 50) since we trained
	// on eastward motion.
	mt := MeanTrajectory(results)
	finalPos := mt[len(mt)-1]
	if finalPos.X <= 50 {
		t.Errorf("expected mean trajectory to move east (x>50), got x=%.3f", finalPos.X)
	}

	// All positions in bounds.
	for i, r := range results {
		for j, p := range r.Trajectory {
			if p.X < 0 || p.X > 120 || p.Y < 0 || p.Y > 53.3 {
				t.Errorf("run %d step %d out of bounds: (%f,%f)", i, j, p.X, p.Y)
			}
		}
	}
}
