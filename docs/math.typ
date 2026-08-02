#set math.equation(numbering: "(1)")
#set page(margin: 50pt)
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
[dot.c]^W : "world" quad ("ArUco marker centre", +x, +y "marker edges", +z "up")\
[dot.c]^B : "body" quad (+x "nose", +y "portside", +z "up")\
[dot.c]^C : "camera" quad (+x "right", +y "down", +z "optical axis")
$

#defs[
  $[a]_times$: skew matrix, $[a]_times b = a times b$\
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
One flat marker fixes the world frame; each camera solves its own pose against it.

$
X_j = (plus.minus L\/2, plus.minus L\/2, 0)^top, quad j = 1 dots 4
$

$
s vec(u, v, 1) = K mat(delim: "[", augment: #1, R, t) vec(X_j, 1)
$

#defs[
  $X_j$: marker corners in $W$ [m] $quad$ $L$: marker edge [m]\
  $x_j$: measured corner pixels [px] $quad$ $K$: intrinsics\
  $R in "SO"(3), t$: world $arrow.r$ camera pose
]

=== Solution

$
hat(R), hat(t) = argmin_(R, t) sum_(j=1)^4 norm(pi(K (R X_j + t)) - x_j)^2
$

Closed form for a planar square (IPPE), which returns both mirror poses; keep the
smaller residual, report the ratio as an ambiguity check.

$
C = -R^top t
$

#defs[
  $C$: camera centre in $W$ [m]
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
A X = 0, quad A = mat(
  u_0 p_0^(3top) - p_0^(1top);
  v_0 p_0^(3top) - p_0^(2top);
  u_1 p_1^(3top) - p_1^(1top);
  v_1 p_1^(3top) - p_1^(2top))
$

#defs[
  $P_i = K mat(delim: "[", augment: #1, R, t)$: projection matrix, camera $i$ $quad$ $p_i^(k top)$: its row $k$\
  $x_0, x_1$: undistorted pixels of one LED [px]\
  $X$: right singular vector of $A$ at $sigma_min$, dehomogenised
]

=== Correspondence
Blobs arrive unlabelled; $3! = 6$ pairings, take the smallest mean reprojection error.

$
sigma^* = argmin_(sigma in S_3) 1/3 sum_(k=1)^3 sum_(i=0)^1
  norm(pi(P_i X_(k,sigma)) - x_(i,k))^2
$

=== Rigid fit (Kabsch)

$
H = sum_(k=1)^3 (b_k - macron(b))(w_k - macron(w))^top = U Sigma V^top
$

$
R = V mat(1,0,0; 0,1,0; 0,0,det(V U^top)) U^top, quad
p = macron(w) - R macron(b)
$

$
"rmsd" = sqrt(1/3 sum_(k=1)^3 norm(R b_k + p - w_k)^2)
$

#defs[
  $b_k$: LED positions in $B$ [m] $quad$ $w_k$: triangulated LEDs in $W$ [m]\
  $det(V U^top)$ term: forces a rotation, not a reflection\
  $"rmsd"$: fit residual [m], also the quality gate\
  labelling of $w_k$ against $b_k$: chosen by the same minimisation
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
One axis, error $e = z - z_"ref"$. Betaflight holds attitude; the commanded lean
arrives through a first-order lag.

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
World demand $arrow.r$ body by measured yaw $psi$, then to sticks.

$
a_"fwd" = a_x cos psi + a_y sin psi, quad
a_"left" = -a_x sin psi + a_y cos psi
$

$
theta = arctan(a_"fwd"\/g), quad
phi = arctan(-a_"left"\/g), quad
"thr" = "ff" (1 + a_z\/g) \/ cos theta
$

#defs[
  $theta, phi$: pitch, roll commands [rad], clamped to $theta_max$\
  $"ff"$: hover feed-forward, learned in flight from the standing vertical integral\
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
Nominal $(p, v, R, omega)$ kept as is; the filter propagates the error state, so
rotation needs no global parameterisation.

$
delta x = vec(delta p, delta v, delta theta, delta omega) in RR^12, quad
R_"true" = R Exp(delta theta)
$

#defs[
  $p, v$: position, velocity in $W$ [m], [m/s]\
  $R, omega$: body $arrow.r$ world rotation, body rate [rad/s]\
  error on the right: keeps $H$ simple, no Euler singularity
]

=== Prediction
Constant velocity, constant body rate; acceleration is process noise.

$
p^- arrow.l p + v Delta t, quad
R^- arrow.l R Exp(omega Delta t), quad
P^- arrow.l F P F^top + Q
$

$
F = mat(
  I, I Delta t, 0, 0;
  0, I, 0, 0;
  0, 0, Exp(omega Delta t)^top, I Delta t;
  0, 0, 0, I)
$

=== Measurement model
Three LEDs, $z in RR^9$.

$
h_k = R b_k + p, quad
H_k = mat(delim: "[", I, 0, -R [b_k]_times, 0)
$

#defs[
  $z$: triangulated LED positions [m], labelled against the *prediction*\
  $R_m = sigma_m^2 I_9$: triangulation noise
]

=== Update
Gate first, then Joseph form.

$
y = z - h, quad S = H P^- H^top + R_m, quad
y^top S^(-1) y <= chi^2_"gate"
$

$
K = P^- H^top S^(-1), quad delta x = K y
$

$
P^+ arrow.l (I - K H) P^- (I - K H)^top + K R_m K^top
$

$
p^+ = p^- + delta p, quad v^+ = v^- + delta v, quad
R^+ = R^- Exp(delta theta), quad omega^+ = omega^- + delta omega
$

#defs[
  $y$: innovation $quad$ $S$: innovation covariance $quad$ $K$: Kalman gain\
  gate rejected, or no measurement: $quad$ predict only, invalid past $t_"coast"$\
  $R^+$: re-orthonormalised by SVD
]
]
