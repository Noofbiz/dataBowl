package simulate

import "math"

const (
	dt        = 0.1
	maxSpeed  = 12.0
	fieldMinX = 0.0
	fieldMaxX = 120.0
	fieldMinY = 0.0
	fieldMaxY = 53.3
)

// Vec2 is a 2D vector in field coordinates.
type Vec2 struct {
	X, Y float64
}

// State is the kinematic state of a simulated particle at one time step.
type State struct {
	Pos   Vec2
	Vel   Vec2 // velocity components (vx, vy)
	Speed float64
	Dir   float64 // degrees, clockwise from +Y
}

// Trajectory is a sequence of positions for one Monte Carlo run.
type Trajectory []Vec2

// Result is the outcome of one Monte Carlo run: the full trajectory.
type Result struct {
	Trajectory Trajectory
}

// Simulator runs Monte Carlo particle simulations using a ForceTable.
type Simulator struct {
	forces *ForceTable
}

// NewSimulator constructs a simulator backed by the provided force table.
func NewSimulator(forces *ForceTable) *Simulator {
	return &Simulator{forces: forces}
}

// Run simulates nRuns independent trajectories for a player starting at
// initState, for nSteps frames (each step = dt = 0.1s).
func (sim *Simulator) Run(role string, initState State, nRuns, nSteps int) []Result {
	if nRuns <= 0 {
		return nil
	}

	results := make([]Result, nRuns)
	for run := 0; run < nRuns; run++ {
		state := normalizeState(initState)
		trajectory := make(Trajectory, 0, max(1, nSteps+1))
		trajectory = append(trajectory, state.Pos)

		for step := 0; step < nSteps; step++ {
			force := Vec2{}
			if sim != nil && sim.forces != nil {
				force = sim.forces.Sample(role, state.Speed, state.Dir)
			}

			state.Vel.X += force.X * dt
			state.Vel.Y += force.Y * dt
			state.Vel = clampVelocity(state.Vel, maxSpeed)

			state.Pos.X = clamp(state.Pos.X+state.Vel.X*dt, fieldMinX, fieldMaxX)
			state.Pos.Y = clamp(state.Pos.Y+state.Vel.Y*dt, fieldMinY, fieldMaxY)

			state.Speed, state.Dir = speedAndDir(state.Vel)
			trajectory = append(trajectory, state.Pos)
		}

		results[run] = Result{Trajectory: trajectory}
	}

	return results
}

// MeanTrajectory averages multiple trajectories into a single mean path.
func MeanTrajectory(results []Result) Trajectory {
	if len(results) == 0 {
		return nil
	}

	maxLen := 0
	for _, result := range results {
		if l := len(result.Trajectory); l > maxLen {
			maxLen = l
		}
	}
	if maxLen == 0 {
		return nil
	}

	mean := make(Trajectory, maxLen)
	counts := make([]int, maxLen)
	for _, result := range results {
		for i, pos := range result.Trajectory {
			mean[i].X += pos.X
			mean[i].Y += pos.Y
			counts[i]++
		}
	}

	for i := range mean {
		if counts[i] == 0 {
			continue
		}
		mean[i].X /= float64(counts[i])
		mean[i].Y /= float64(counts[i])
	}

	return mean
}

func normalizeState(state State) State {
	speed, dir := speedAndDir(state.Vel)
	if speed > 0 {
		state.Speed = speed
		state.Dir = dir
		return state
	}

	if state.Speed > 0 {
		state.Vel = velocityFromPolar(state.Speed, state.Dir)
		state.Speed, state.Dir = speedAndDir(state.Vel)
		return state
	}

	state.Vel = Vec2{}
	state.Speed = 0
	state.Dir = normalizeDegrees(state.Dir)
	return state
}

func velocityFromPolar(speed, dir float64) Vec2 {
	radians := dir * math.Pi / 180.0
	return Vec2{
		X: speed * math.Sin(radians),
		Y: speed * math.Cos(radians),
	}
}

func speedAndDir(vel Vec2) (float64, float64) {
	speed := math.Hypot(vel.X, vel.Y)
	if speed == 0 {
		return 0, 0
	}
	return speed, normalizeDegrees(math.Atan2(vel.X, vel.Y) * 180.0 / math.Pi)
}

func clampVelocity(vel Vec2, max float64) Vec2 {
	speed := math.Hypot(vel.X, vel.Y)
	if speed == 0 || speed <= max {
		return vel
	}
	scale := max / speed
	return Vec2{X: vel.X * scale, Y: vel.Y * scale}
}

func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}

func normalizeDegrees(dir float64) float64 {
	dir = math.Mod(dir, 360)
	if dir < 0 {
		dir += 360
	}
	return dir
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
