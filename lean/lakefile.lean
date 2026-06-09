import Lake
open Lake DSL

package emergo_convergence where
  name := "emergo_convergence"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

lean_lib EmergoConvergence where
  roots := #[`EmergoConvergence]
