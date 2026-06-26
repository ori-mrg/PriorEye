rclone config
DATASET_DIR="/dataset"
mkdir -p $DATASET_DIR


echo "🚀 Downloading embedding, gsv_metadata, nuplan-maps-v1.1.zip, openscene_metadata_trainval.tgz"
rclone copy -vvP gcloud:memory_embedding/ .

tar -xf embedding_nuplan.tar && rm embedding_nuplan.tar
mv embedding $DATASET_DIR/
echo "✅ Embedding"

mkdir -p $DATASET_DIR/gsv
mv gsv_metadata.pkl $DATASET_DIR/gsv/
echo "✅ GSV Metadata"

echo "🗺️ Maps & Metadata ..."
unzip -qo nuplan-maps-v1.1.zip && rm nuplan-maps-v1.1.zip
mv nuplan-maps-v1.0 $DATASET_DIR/maps
echo "✅ Maps"

tar -xzf openscene_metadata_trainval.tgz
echo "✅ Metadata"


echo "📸 Camera Loop ..."
mkdir -p $DATASET_DIR/sensor_blobs/trainval
mkdir -p $DATASET_DIR/navsim_logs/trainval

seq 0 199 | xargs -P 16 -I {} sh -c '
    FILE="cam_{}.tgz"
    URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{}.tgz"
    
    # 1. Download file (using resume -c option)
    wget -c -q -O $FILE $URL
    
    # 2. Extract (print message on error)
    if tar -xzf $FILE; then
        echo "✅ Cam {} Done"
        rm $FILE
    else
        echo "Cam {} error - file may be corrupted."
    fi
'


echo "📂 Organizing folders"

if [ -d "openscene-v1.1" ]; then
            mv openscene-v1.1/sensor_blobs/trainval/* $DATASET_DIR/sensor_blobs/trainval/ 2>/dev/null
            mv openscene-v1.1/meta_datas/trainval/* $DATASET_DIR/navsim_logs/trainval/ 2>/dev/null
                rm -rf openscene-v1.1
fi


echo “Downloading.. traj_pdm_v2
mkdir -p $DATASET_DIR/traj_pdm_v2/ori
mkdir -p $DATASET_DIR/traj_pdm_v2/random_aug


rclone copy gcloud:traj_pdm_v2/ori/navtrain_16384.pkl $DATASET_DIR/traj_pdm_v2/ori/ -P
rclone copy gcloud:traj_pdm_v2/ori/navtrain_16384.pkl  $DATASET_DIR/traj_pdm_v2/ori -P


echo “Downloading.. models”
mkdir -p $DATASET_DIR/models
wget -q -P $DATASET_DIR/models https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth 

