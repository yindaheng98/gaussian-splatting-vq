# !/bin/bash
COLMAP_EXE_PATH=colmap
dense() {
    cp data/$1/frame1/sparse/0/*.bin data/$1/frame1/sparse
    if [ ! -e "data/$1/frame1/fused.ply" ]; then
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
    fi
    if [ ! -e "data/$1/frame1/meshed-poisson.ply" ]; then
    # echo \
    $COLMAP_EXE_PATH poisson_mesher \
        --input_path data/$1/frame1/fused.ply \
        --output_path data/$1/frame1/meshed-poisson.ply
    fi
    # echo \
    if [ ! -e "data/$1/frame1/meshed-delaunay.ply" ]; then
    $COLMAP_EXE_PATH delaunay_mesher \
        --input_path data/$1/frame1 \
        --output_path data/$1/frame1/meshed-delaunay.ply
    fi
    if [ ! -e "data/$1/frame1/colorful-delaunay.ply" ]; then
    # echo \
    python dataset_convert/delaunay2ply.py \
        --delaunay data/$1/frame1/meshed-delaunay.ply \
        --reference data/$1/frame1/meshed-poisson.ply \
        --save data/$1/frame1/colorful-delaunay.ply \
        --batch 512
    fi
    if [ ! -e "data/$1/frame1/filtered-poisson.ply" ]; then
    # echo \
    python dataset_convert/poisson_filter.py \
        --poisson data/$1/frame1/meshed-poisson.ply \
        --reference data/$1/frame1/colorful-delaunay.ply \
        --save data/$1/frame1/filtered-poisson.ply
    fi
    rm data/$1/frame1/sparse/0/points3D.ply
    cp data/$1/frame1/filtered-poisson.ply data/$1/frame1/sparse/0/points3D.ply
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

dense taekwondo
dense walking
