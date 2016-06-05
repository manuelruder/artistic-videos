set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}


# Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ]; then
   echo "Usage: ./stylizeVideo <path_to_video> <path_to_style_image>"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_image=$2

# Create output folder
mkdir -p $filename


echo ""
read -p "Which backend do you want to use? \
For Nvidia GPU, use cudnn if available, otherwise nn. \
For non-Nvidia GPU, use clnn. Note: You have to have the given backend installed in order to use it. [nn] $cr > " backend
backend=${backend:-nn}

if [ "$backend" == "cudnn" ]; then
  echo ""
  read -p "This algorithm needs a lot of memory. \
  For a resolution of 450x350 you'll need roughly 2GB VRAM. \
  VRAM usage increases linear with resolution. \
  Please enter a resolution at which the video should be processed, \
  in the format w:h, or leave blank to use the original resolution $cr > " resolution
elif [ "$backend" = "nn" ] || [ "$backend" = "clnn" ]; then
  echo ""
  read -p "This algorithm needs a lot of memory. \
  For a resolution of 450x350 you'll need roughly 4GB VRAM. \
  VRAM usage increases linear with resolution. \
  Maximum recommended resolution with a Titan X 12GB: 960:540. \
  Please enter a resolution at which the video should be processed, \
  in the format w:h, or leave blank to use the original resolution $cr > " resolution
else
  echo "Unknown backend."
  exit 1
fi

# Save frames of the video as individual image files
if [ -z $resolution ]; then
  $FFMPEG -i $1 ${filename}/frame_%04d.ppm
  resolution=default
else
  $FFMPEG -i $1 -vf scale=$resolution ${filename}/frame_%04d.ppm
fi

echo ""
read -p "How much do you want to weight the style reconstruction term? \
Default value: 1e2 for a resolution of 450x350. Increase for a higher resolution. \
[1e2] $cr > " style_weight
style_weight=${style_weight:-1e2}

temporal_weight=1e3

echo ""
read -p "Enter the zero-indexed ID of the GPU to use, or -1 for CPU mode (very slow!).\
 [0] $cr > " gpu
gpu=${gpu:-0}

echo ""
echo "Computing optical flow. This may take a while..."
bash makeOptFlow.sh ./${filename}/frame_%04d.ppm ./${filename}/flow_$resolution

# Perform style transfer
th artistic_video.lua \
-content_pattern ${filename}/frame_%04d.ppm \
-flow_pattern ${filename}/flow_${resolution}/backward_[%d]_{%d}.flo \
-flowWeight_pattern ${filename}/flow_${resolution}/reliable_[%d]_{%d}.pgm \
-style_weight $style_weight \
-temporal_weight $temporal_weight \
-output_folder ${filename}/ \
-style_image $style_image \
-backend $backend \
-gpu $gpu \
-cudnn_autotune \
-number_format %04d

# Create video from output images.
$FFMPEG -i ${filename}/out-%04d.png ${filename}-stylized.$extension