package simulate

import (
	"math"
	rand "math/rand/v2"
	"sort"
	"strconv"

	"github.com/Noofbiz/dataBowl/datasets"
)

const anyBucket = -1

type forceKey struct {
	role        string
	speedBucket int
	dirBucket   int
}

// ForceTable stores empirical acceleration samples bucketed by role, speed,
// and direction.
type ForceTable struct {
	samples map[forceKey][]Vec2
	rng     *rand.Rand
}

// BuildForceTable scans every consecutive frame pair across the supplied games
// and collects empirical acceleration samples for each observed player.
func BuildForceTable(games []*datasets.Game, seed int64) *ForceTable {
	ft := &ForceTable{
		samples: make(map[forceKey][]Vec2),
		rng:     rand.New(rand.NewPCG(uint64(seed), 0)),
	}

	for _, game := range games {
		if game == nil {
			continue
		}
		for _, play := range game.Plays {
			frameIDs := sortedFrameIDs(play.Frames)
			for i := 0; i+1 < len(frameIDs); i++ {
				current := play.Frames[frameIDs[i]]
				next := play.Frames[frameIDs[i+1]]
				for playerID, player := range current.Players {
					nextPlayer, ok := next.Players[playerID]
					if !ok {
						continue
					}

					currentVel := velocityFromPolar(player.Velocity, player.AngleOfMomentum)
					nextVel := velocityFromPolar(nextPlayer.Velocity, nextPlayer.AngleOfMomentum)
					sample := Vec2{
						X: (nextVel.X - currentVel.X) / dt,
						Y: (nextVel.Y - currentVel.Y) / dt,
					}

					role := player.PlayerRole
					if nextPlayer.PlayerRole != "" {
						role = nextPlayer.PlayerRole
					}

					speedBucket := bucketSpeed(player.Velocity)
					dirBucket := bucketDir(player.AngleOfMomentum)
					ft.addSample(role, speedBucket, dirBucket, sample)
				}
			}
		}
	}

	return ft
}

// Sample draws one empirical acceleration sample for the closest matching
// bucket. It falls back from exact role/speed/direction matches to broader
// aggregates, returning the zero vector if no samples exist.
func (ft *ForceTable) Sample(role string, speed, dir float64) Vec2 {
	if ft == nil || len(ft.samples) == 0 || ft.rng == nil {
		return Vec2{}
	}

	speedBucket := bucketSpeed(speed)
	dirBucket := bucketDir(dir)
	keys := [...]forceKey{
		{role: role, speedBucket: speedBucket, dirBucket: dirBucket},
		{role: role, speedBucket: speedBucket, dirBucket: anyBucket},
		{role: role, speedBucket: anyBucket, dirBucket: anyBucket},
		{role: "", speedBucket: anyBucket, dirBucket: anyBucket},
	}

	for _, key := range keys {
		samples := ft.samples[key]
		if len(samples) == 0 {
			continue
		}
		return samples[ft.rng.IntN(len(samples))]
	}

	return Vec2{}
}

func (ft *ForceTable) addSample(role string, speedBucket, dirBucket int, sample Vec2) {
	keys := [...]forceKey{
		{role: role, speedBucket: speedBucket, dirBucket: dirBucket},
		{role: role, speedBucket: speedBucket, dirBucket: anyBucket},
		{role: role, speedBucket: anyBucket, dirBucket: anyBucket},
		{role: "", speedBucket: speedBucket, dirBucket: dirBucket},
		{role: "", speedBucket: speedBucket, dirBucket: anyBucket},
		{role: "", speedBucket: anyBucket, dirBucket: anyBucket},
	}
	for _, key := range keys {
		ft.samples[key] = append(ft.samples[key], sample)
	}
}

func sortedFrameIDs(frames map[string]datasets.Frame) []string {
	ids := make([]string, 0, len(frames))
	for id := range frames {
		ids = append(ids, id)
	}

	sort.Slice(ids, func(i, j int) bool {
		left, leftErr := strconv.Atoi(ids[i])
		right, rightErr := strconv.Atoi(ids[j])
		switch {
		case leftErr == nil && rightErr == nil:
			return left < right
		case leftErr == nil:
			return true
		case rightErr == nil:
			return false
		default:
			return ids[i] < ids[j]
		}
	})

	return ids
}

func bucketSpeed(speed float64) int {
	if speed < 0 {
		speed = 0
	}
	return int(speed / 2.0)
}

func bucketDir(dir float64) int {
	dir = normalizeDegrees(dir)
	return int(math.Floor(dir/45.0)) % 8
}
