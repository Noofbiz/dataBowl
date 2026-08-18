package datasets

// Game represents an NFL game. Fields use the Kaggle CSV naming and types
// where IDs are strings and enums are kept as strings to match the raw data.
type Game struct {
	ID                string `csv:"game_id"` // game_id (string)
	Plays             map[string]Play
	HomeTeam          string `csv:"home_team"`        // home team abbreviation
	VisitorTeam       string `csv:"visitor_team"`     // visitor team abbreviation
	Season            int    `csv:"season"`           // numeric season year
	Date              string `csv:"game_date"`        // game date (YYYY-MM-DD)
	TimeEst           string `csv:"kickoff_time_est"` // kickoff time (EST) as string
	HomeFinalScore    int    `csv:"home_final_score"`
	VisitorFinalScore int    `csv:"visitor_final_score"`
}

// Play represents a single play within a game.
type Play struct {
	ID     string `csv:"play_id"` // play_id
	Frames map[string]Frame

	// player_to_predict: the nfl_id (string) of the player to predict (if present)
	PlayerToPredictNFLID string `csv:"nfl_id"`
	PlayerToPredict      bool   `csv:"player_to_predict"`

	PlayDirection                    string  `csv:"play_direction"`
	AbsoluteYardlineNumber           float64 `csv:"absolute_yardline_number"`
	BallLandX                        float64 `csv:"ball_land_x"`
	BallLandY                        float64 `csv:"ball_land_y"`
	TextDescription                  string  `csv:"text_description"`
	Quarter                          int     `csv:"quarter"`
	Down                             int     `csv:"down"`
	GameClock                        string  `csv:"game_clock"` // e.g. "12:34"
	YardsToGo                        float64 `csv:"yards_to_go"`
	Offense                          string  `csv:"offense"`
	Defense                          string  `csv:"defense"`
	YardlineSide                     string  `csv:"yardline_side"`
	YardlineNumber                   float64 `csv:"yardline_number"`
	PreSnapHomeScore                 int     `csv:"pre_snap_home_score"`
	PreSnapVisitorScore              int     `csv:"pre_snap_visitor_score"`
	PassResult                       string  `csv:"pass_result"`
	Penalty                          bool    `csv:"penalty"`
	PassLength                       float64 `csv:"pass_length"`
	OffenseFormation                 string  `csv:"offense_formation"`
	ReceiverAlignment                string  `csv:"receiver_alignment"`
	PlayAction                       bool    `csv:"play_action"`
	TargetedReceiverRoute            string  `csv:"targeted_receiver_route"`
	Dropback                         string  `csv:"dropback"`
	DropbackDistance                 float64 `csv:"dropback_distance"`
	PassLocation                     string  `csv:"pass_location"`
	DefendersInTheBox                float64 `csv:"defenders_in_the_box"`
	TeamCoverageManZone              string  `csv:"team_coverage_man_zone"`
	TeamCoverage                     string  `csv:"team_coverage"`
	PenaltyYards                     float64 `csv:"penalty_yards"`
	PrePenaltyYardsGained            float64 `csv:"pre_penalty_yards_gained"`
	YardsGained                      float64 `csv:"yards_gained"`
	ExpectedPoints                   float64 `csv:"expected_points"`
	ExpectedPointsAdded              float64 `csv:"expected_points_added"`
	PreSnapHomeTeamWinProbability    float64 `csv:"pre_snap_home_team_win_probability"`
	PreSnapVisitorTeamWinProbability float64 `csv:"pre_snap_visitor_team_win_probability"`
	HomeTeamWinProbabilityAdded      float64 `csv:"home_team_win_probability_added"`
	VisitorTeamWinProbabilityAdded   float64 `csv:"visitor_team_win_probability_added"`
}

// Frame represents the frozen tracking snapshot at a frame_id within a play.
type Frame struct {
	ID          string `csv:"frame_id"` // frame_id as string
	Players     map[string]Player
	BallX       float64 `csv:"ball_land_x"`
	BallY       float64 `csv:"ball_land_y"`
	BallVisible bool    `csv:"ball_visible"`
}

// Player represents an observed player (or the ball) in a frame.
// Fields align with the Kaggle tracking CSV: nfl_id, display_name, jersey_number,
// x,y,s,a,o,dir,player_side,player_role,player_to_predict.
type Player struct {
	ID           string `csv:"nfl_id"` // nfl_id (may be empty for the ball)
	DisplayName  string `csv:"player_name"`
	JerseyNumber string `csv:"jersey_number"`

	Team            string `csv:"team"`        // team abbreviation
	PlayerSide      string `csv:"player_side"` // "Offense" or "Defense"
	PlayerRole      string `csv:"player_role"` // e.g. "Passer", "Targeted Receiver"
	PlayerToPredict bool   `csv:"player_to_predict"`

	// Optional roster fields (some datasets include these in players.csv)
	Height    string  `csv:"player_height"` // e.g. "6-2"
	Weight    float64 `csv:"player_weight"`
	BirthDate string  `csv:"player_birth_date"` // YYYY-MM-DD

	TypicalRole string `csv:"player_position"`
	PlayRole    string `csv:"player_role"`

	X                float64 `csv:"x"`
	Y                float64 `csv:"y"`
	Velocity         float64 `csv:"s"` // alias for S
	Acceleration     float64 `csv:"a"` // alias for A
	Orientation      float64 `csv:"o"` // alias for O
	AngleOfMomentum  float64 `csv:"dir"`
	OutputFrameCount uint    `csv:"num_frames_output"`
}
