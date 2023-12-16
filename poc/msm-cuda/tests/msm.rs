// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{G1Affine, G2Affine};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{G1Affine, G2Affine};
#[cfg(feature = "bn254")]
use ark_bn254::G1Affine;
use ark_ec::msm::VariableBaseMSM;
use ark_ec::ProjectiveCurve;
use ark_ff::BigInteger256;
use ark_std::test_rng;
use halo2_proofs::arithmetic::best_multiexp;
use halo2_proofs::arithmetic::CurveExt;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256;
use halo2_proofs::halo2curves::group::ff::PrimeField;
use halo2_proofs::halo2curves::group::Curve;
use std::ops::Mul;
use std::str::FromStr;

use msm_cuda::*;

// #[test]
// fn msm_correctness() {
//     let test_npow = std::env::var("TEST_NPOW").unwrap_or("15".to_string());
//     let npoints_npow = i32::from_str(&test_npow).unwrap();

//     let (points, scalars) =
//         util::generate_points_scalars::<G1Affine>(1usize << npoints_npow);

//     let msm_result = multi_scalar_mult_arkworks(points.as_slice(), unsafe {
//         std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
//     })
//     .into_affine();

//     let arkworks_result =
//         VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
//             std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
//         })
//         .into_affine();

//     assert_eq!(msm_result, arkworks_result);
// }

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[test]
fn msm_fp2_correctness() {
    let test_npow = std::env::var("TEST_NPOW").unwrap_or("14".to_string());
    let npoints_npow = i32::from_str(&test_npow).unwrap();

    let (points, scalars) =
        util::generate_points_scalars::<G2Affine>(1usize << npoints_npow);

    let msm_result =
        multi_scalar_mult_fp2_arkworks(points.as_slice(), unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        })
        .into_affine();

    let arkworks_result =
        VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        })
        .into_affine();

    assert_eq!(msm_result, arkworks_result);
}

#[test]
fn msm_correctness_halo2() {
    let mut rng = test_rng();

    let test_npow = std::env::var("TEST_NPOW").unwrap_or("15".to_string());
    let npoints_npow = i32::from_str(&test_npow).unwrap();
    let size = 1usize << npoints_npow;
    
    let points = (0..size)
        .map(|_| bn256::G1Affine::random(&mut rng))
        .collect::<Vec<_>>();
    let scalars = (0..size)
        .map(|_| bn256::Fr::random(&mut rng))
        .collect::<Vec<_>>();
    let scalars_repr = scalars.iter().map(|x| x.to_repr()).collect::<Vec<_>>();

    let msm_result_jacobian =
        multi_scalar_mult_halo2(points.as_slice(), scalars_repr.as_slice());

    let msm_result = bn256::G1::new_jacobian(
        msm_result_jacobian.x,
        msm_result_jacobian.y,
        msm_result_jacobian.z,
    )
    .unwrap()
    .to_affine();

    let halo2_result = best_multiexp(&scalars, &points).to_affine();

    assert_eq!(msm_result, halo2_result);
}
