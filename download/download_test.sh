DATASET_DIR="/dataset"

wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

echo "📸 Camera Loop ..."
mkdir -p $DATASET_DIR/sensor_blobs/test
mkdir -p $DATASET_DIR/navsim_logs/test

seq 0 31 | xargs -P 16 -I {} sh -c '
    FILE="openscene_sensor_test_camera_{}.tgz"
    URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_{}.tgz"

    wget -c -q -O $FILE $URL

    if tar -xzf $FILE; then
        echo "✅ Cam {} Done"
        rm $FILE
    else
        echo "Cam {} error - file may be corrupted."
    fi
'

echo "📂 Organizing folders"

if [ -d "openscene-v1.1" ]; then
    mv openscene-v1.1/sensor_blobs/test/* $DATASET_DIR/sensor_blobs/test/ 2>/dev/null
    mv openscene-v1.1/meta_datas/test/* $DATASET_DIR/navsim_logs/test/ 2>/dev/null
    rm -rf openscene-v1.1
fi
