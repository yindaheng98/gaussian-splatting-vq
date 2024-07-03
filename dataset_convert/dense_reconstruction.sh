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
        --input_type geometric \
        --output_path data/$1/frame1/fused.ply
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
