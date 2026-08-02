#set math.equation(numbering: "(1)")
#set page(margin: 44pt)
#set text(size: 11pt)

#let defs(body) = block(
  inset: (left: 1.3em), above: 0.55em, below: 1.0em,
  text(size: 8.8pt, fill: rgb("#333333"), body))
#let darkgreen = rgb("#008000")
#let purple = rgb("#6a2c9e")
#let argmin = math.op("arg min", limits: true)
#let Exp = math.op("Exp")

#block(
  stroke: 1pt + red,
  radius: 4pt,
  inset: 10pt,
  fill: gray.lighten(80%)
)[
= Two-Camera IR Mocap Hover
#v(0.3cm)

=== Frames

$
[dot.c]^W : "world" quad ("ArUco marker center", +x, +y "marker edges", +z "up")\
[dot.c]^B : "body" quad (+x "nose", +y "portside", +z "up")\
[dot.c]^(C_i) : "camera" i quad (+x "right", +y "down", +z "optical axis")
$

#defs[
  $[T]^(Y X)$: rotation coordinatizing an $X$-frame tensor in $Y$, $[a]^Y = [T]^(Y X) [a]^X$\
  $[T]^(X Y) = ([T]^(Y X))^top$ $quad$ $[a]_times$: skew matrix, $[a]_times b = a times b$\
  $Exp(phi.alt)$: rotation of angle $norm(phi.alt)$ about $phi.alt \/ norm(phi.alt)$ (Rodrigues)\
  $pi(dot.c)$: perspective divide, $pi(vec(a,b,c)) = vec(a\/c, b\/c)$
]
]

#block(
  stroke: 1pt + darkgreen,
  radius: 4pt,
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== Camera pose

=== Problem
One flat marker fixes $W$; each camera solves its own pose against it.

$
[X_j]^W = (plus.minus L\/2, plus.minus L\/2, 0)^top, quad j = 1 dots 4
$

$
s vec(u, v, 1)_j = K ([T]^(C W) [X_j]^W + [t]^C)
$

#defs[
  $[X_j]^W$: marker corners [m] $quad$ $L$: marker edge [m]\
  $u_j, v_j$: measured corner pixels [px] $quad$ $K$: intrinsics\
  $[T]^(C W), [t]^C$: pose of $W$ as seen from $C$
]

=== Solution

$
[hat(T)]^(C W), [hat(t)]^C = argmin_(T, t)
  sum_(j=1)^4 norm(pi(K ([T]^(C W) [X_j]^W + [t]^C)) - vec(u,v)_j)^2
$

Closed form for a planar square (IPPE), which returns both mirror poses; keep the
smaller residual, report the ratio as an ambiguity check.

$
[c]^W = -[T]^(W C) [t]^C
$

#defs[
  $[c]^W$: camera center [m]
]
]

#block(
  stroke: 1pt + darkgreen,
  radius: 4pt,
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== Drone pose

=== Triangulation
Per LED, from $x times P X = 0$, two rows per camera.

$
A [X]^W = 0, quad A = mat(
  u_0 p_0^(3top) - p_0^(1top);
  v_0 p_0^(3top) - p_0^(2top);
  u_1 p_1^(3top) - p_1^(1top);
  v_1 p_1^(3top) - p_1^(2top))
$

#defs[
  $P_i = K mat(delim: "[", augment: #1, [T]^(C_i W), [t]^(C_i))$: projection matrix $quad$ $p_i^(k top)$: its row $k$\
  $u_i, v_i$: undistorted pixels of one LED in camera $i$ [px]\
  $[X]^W$: right singular vector of $A$ at $sigma_min$, dehomogenized [m]
]

=== Correspondence
Blobs arrive unlabeled; $3! = 6$ pairings, take the smallest mean reprojection error.

$
sigma^* = argmin_(sigma in S_3) 1/3 sum_(k=1)^3 sum_(i=0)^1
  norm(pi(P_i [X_(k,sigma)]^W) - vec(u,v)_(i,k))^2
$

=== Rigid fit (Kabsch)

$
H = sum_(k=1)^3 ([b_k]^B - [macron(b)]^B)([w_k]^W - [macron(w)]^W)^top = U Sigma V^top
$

$
[T]^(W B) = V mat(1,0,0; 0,1,0; 0,0,det(V U^top)) U^top, quad
[p]^W = [macron(w)]^W - [T]^(W B) [macron(b)]^B
$

$
"rmsd" = sqrt(1/3 sum_(k=1)^3 norm([T]^(W B) [b_k]^B + [p]^W - [w_k]^W)^2)
$

#defs[
  $[b_k]^B$: LED positions on the airframe [m] $quad$ $[w_k]^W$: triangulated LEDs [m]\
  $[T]^(W B), [p]^W$: drone attitude and position\
  $det(V U^top)$ term: forces a rotation, not a reflection\
  rmsd: fit residual [m], also the quality gate\
  labeling of $[w_k]^W$ against $[b_k]^B$: chosen by the same minimization
]
]

#block(
  stroke: 1pt + purple,
  radius: 4pt,
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== LQR control law

=== Problem
One axis of $[dot.c]^W$, error $e = z - z_"ref"$. Betaflight holds attitude; the
commanded lean arrives through a first-order lag.

$
x = vec(integral e, e, dot(e), a), quad u = a_"cmd", quad dot(x) = A x + B u
$

$
A = mat(0,1,0,0; 0,0,1,0; 0,0,0,1; 0,0,0,-1\/tau), quad
B = vec(0,0,0,1\/tau)
$

$
J = sum_(k=0)^oo x_k^top Q x_k + u_k^top R u_k
$

#defs[
  $a$: acceleration actually delivered, chasing $u$ [m/s$""^2$]\
  $tau$: attitude (or throttle) lag $+$ transport delay [s]\
  $Q = "diag"(1\/x_(i,max)^2)$, $R = rho^2\/a_max^2$: Bryson weights\
  $rho$: `effort`, trades aggression against smoothness
]

=== Solution
Zero-order hold at the loop rate, then the discrete Riccati equation.

$
mat(A_d, B_d; 0, I) = exp(mat(A, B; 0, 0) Delta t)
$

$
P = A_d^top P A_d - A_d^top P B_d (R + B_d^top P B_d)^(-1) B_d^top P A_d + Q
$

$
K = (R + B_d^top P B_d)^(-1) B_d^top P A_d, quad u = -K x
$

#defs[
  $P$: solved by iterating the recursion to convergence (no SciPy)
]

=== Command mapping
Coordinatize the demand in $B$ using the estimated yaw $psi$, then map to sticks.

$
[a]^B = vec(a_"fwd", a_"left", a_z) = [T]^(B W) [a]^W, quad
[T]^(B W) = mat(cos psi, sin psi, 0; -sin psi, cos psi, 0; 0,0,1)
$

$
theta = arctan(a_"fwd"\/g), quad
phi = arctan(-a_"left"\/g), quad
"thr" = "ff" (1 + a_z\/g) \/ cos theta
$

#defs[
  $theta, phi$: pitch, roll commands [rad], clamped to $theta_max$\
  ff: hover feed-forward, learned in flight from the standing vertical integral\
  yaw: left to the flight controller
]
]

#block(
  stroke: 1pt + darkgreen,
  radius: 4pt,
  inset: 10pt,
  fill: gray.lighten(80%)
)[
== EKF

=== State
Nominal kept as is; the filter propagates the error state, so the rotation needs
no global parameterization.

$
delta x = vec([delta p]^W, [delta v]^W, [delta theta]^B, [delta omega]^B) in RR^12, quad
[T]^(W B)_"true" = [T]^(W B) Exp([delta theta]^B)
$

#defs[
  $[p]^W, [v]^W$: position, velocity [m], [m/s]\
  $[T]^(W B), [omega]^B$: attitude, body rate [rad/s]\
  error on the right: acts in $B$, keeps $H$ simple, no Euler singularity
]

=== Prediction
Constant velocity, constant body rate; acceleration is process noise.

$
[p]^W arrow.l [p]^W + [v]^W Delta t, quad
[T]^(W B) arrow.l [T]^(W B) Exp([omega]^B Delta t), quad
P arrow.l F P F^top + Q
$

$
F = mat(
  I, I Delta t, 0, 0;
  0, I, 0, 0;
  0, 0, Exp([omega]^B Delta t)^top, I Delta t;
  0, 0, 0, I)
$

=== Measurement model
Three LEDs, $z in RR^9$.

$
h_k = [T]^(W B) [b_k]^B + [p]^W, quad
H_k = mat(delim: "[", I, 0, -[T]^(W B) [ [b_k]^B ]_times, 0)
$

#defs[
  $z$: triangulated $[w_k]^W$ [m], labeled against the *prediction*\
  $R_m = sigma_m^2 I_9$: triangulation noise
]

=== Update
Gate first, then Joseph form.

$
y = z - h, quad S = H P^- H^top + R_m, quad
y^top S^(-1) y <= chi^2_"gate"
$

$
K = P^- H^top S^(-1), quad delta x = K y, quad
P^+ arrow.l (I - K H) P^- (I - K H)^top + K R_m K^top
$

$
[p]^(W+) = [p]^(W-) + [delta p]^W, quad
[T]^(W B+) = [T]^(W B-) Exp([delta theta]^B)
$

#defs[
  $y$: innovation $quad$ $S$: innovation covariance $quad$ $K$: Kalman gain\
  gate rejected, or no measurement: $quad$ predict only, invalid past $t_"coast"$\
  $[T]^(W B+)$: re-orthonormalized by SVD
]
]
