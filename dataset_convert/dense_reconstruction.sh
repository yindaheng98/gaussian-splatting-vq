# !/bin/bash
COLMAP_EXE_PATH=colmap
dense() {
    cp data/$1/frame1/sparse/0/* data/$1/frame1/sparse
    # echo \
    $COLMAP_EXE_PATH patch_match_stereo \
        --workspace_path data/$1/frame1 \
        --workspace_format COLMAP \
        --PatchMatchStereo.max_image_size 2000 \
        --PatchMatchStereo.geom_consistency true
    # echo \
    $COLMAP_EXE_PATH stereo_fusion \
        --workspace_path data/$1/frame1 \
        --workspace_format COLMAP \
        --input_type photometric \
        --output_path data/$1/frame1/fused.ply
    # echo \
    $COLMAP_EXE_PATH poisson_mesher \
        --input_path data/$1/frame1/fused.ply \
        --output_path data/$1/frame1/meshed-poisson.ply
    # echo \
    $COLMAP_EXE_PATH delaunay_mesher \
        --input_path data/$1/frame1 \
        --output_path data/$1/frame1/meshed-delaunay.ply
}

dense coffee_martini
dense cook_spinach
dense cut_roasted_beef
dense flame_salmon_1
dense flame_steak
dense sear_steak

dense discussion
dense stepin
dense trimming
dense vrheadset
