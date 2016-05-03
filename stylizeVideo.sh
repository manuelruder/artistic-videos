set -e

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
read -p "This algorithm needs a lot of memory. \
For a resolution of 450x350 you'll need roughly 4GB VRAM. \
VRAM usage increases linear with resolution. \
For HD resolution, you would probably need a Titan X 12GB. \
Please enter a resolution at which the video should be processed, \
in the format w:h, or leave blank to use the original resolution : " resolution

# Save frames of the video as individual image files
if [ -z $resolution ]; then
  $FFMPEG -i $1 ${filename}/frame_%04d.ppm
  resolution=default
else
  $FFMPEG -i $1 -vf scale=$resolution ${filename}/frame_%04d.ppm
fi

echo ""
echo "Computing optical flow. This may take a while..."
./makeOptFlow.sh ./${filename}/frame_%04d.ppm ./${filename}/flow_$resolution

echo ""
read -p "How much do you want to weight the style reconstruction term? \
Default value: 1e2 for a resolution of 450x350. Increase for a higher resolution. \
[1e2] : " style_weight
style_weight=${style_weight:-1e2}

temporal_weight=1e3

echo ""
read -p "Enter the zero-indexed ID of the GPU to use, or -1 for CPU mode (very slow!).\
 [0] : " gpu
gpu=${gpu:-0}

# Perform style transfer
th artistic_video.lua \
-content_pattern ${filename}/frame_%04d.ppm \
-flow_pattern ${filename}/flow_${resolution}/backward_[%d]_{%d}.flo \
-flowWeight_pattern ${filename}/flow_${resolution}/reliable_[%d]_{%d}.pgm \
-style_weight $style_weight \
-temporal_weight $temporal_weight \
-output_folder ${filename}/ \
-style_image $style_image \
-gpu $gpu

# Create video from output images.
$FFMPEG -i ${filename}/out-%d.png ${filename}-stylized.$extension