# Copyright 2016, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any commercial
# use must be negotiated with the Office of Technology Transfer at the
# California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.

uint64 timestamp          # timestamp when the trajectory was generated
uint16 n_segs # number of segments
float32[] x_coeffs # x polynomial coefficients, segment by segment, starting at highest order coefficient
float32[] y_coeffs # y polynomial coefficients, segment by segment, starting at highest order coefficient
float32[] z_coeffs # z polynomial coefficients, segment by segment, starting at highest order coefficient
float32[] yaw_coeffs # yaw polynomial coefficients, segment by segment, starting at highest order coefficient
float32[] t_trans # transition times, segment by segment
uint8 n_x_coeffs # number of coefficients per segment for x
uint8 n_y_coeffs # number of coefficients per segment for y
uint8 n_z_coeffs # number of coefficients per segment for z
uint8 n_yaw_coeffs # number of coefficients per segment for yaw
