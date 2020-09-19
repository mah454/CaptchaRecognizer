/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ir.moke.captcha;

import net.sourceforge.tess4j.Tesseract;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.*;

public class CaptchaReader {
    static {
        OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private static final String srcImagePath = "captcha/c1.jpeg";

    public static void main(String[] args) throws Exception {
        BufferedImage sourceBufferedImage = ImageIO.read(new File(srcImagePath));
        List<BufferedImage> bufferedImageList = normalizeCaptcha(sourceBufferedImage);
        String code = getCode(bufferedImageList);
        System.out.println(code);
    }

    public static Mat bufferedImageToMat(BufferedImage bufferedImage) {
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), CvType.CV_8UC3);
        DataBuffer dataBuffer = bufferedImage.getRaster().getDataBuffer();
        byte[] data = ((DataBufferByte) dataBuffer).getData();
        mat.put(0, 0, data);
        return mat;
    }

    private static BufferedImage matToBufferedImage(Mat invert) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", invert, matOfByte);
        byte[] bytes = matOfByte.toArray();
        try {
            return ImageIO.read(new ByteArrayInputStream(bytes));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static List<BufferedImage> normalizeCaptcha(BufferedImage bufferedImage) {
        List<BufferedImage> bufferedImageList = new ArrayList<>();
        Mat source = bufferedImageToMat(bufferedImage);

        Mat hsvImage = new Mat();
        cvtColor(source, hsvImage, Imgproc.COLOR_BGR2GRAY);

        Mat gaussianBlur = new Mat();
        GaussianBlur(hsvImage, gaussianBlur, new Size(1, 11), 0);

        Mat medianBlur = new Mat();
        medianBlur(gaussianBlur, medianBlur, 5);

        Mat threshold = new Mat();
        threshold(medianBlur, threshold, 200, 255, Imgproc.THRESH_BINARY_INV);

        Mat invert = new Mat();
        Core.bitwise_not(threshold, invert);


        Mat canny = new Mat();
        Canny(invert, canny, 170, 250);

        List<MatOfPoint> contours = findContours(canny);

        contours.sort((o1, o2) -> {
            Rect rect1 = Imgproc.boundingRect(o1);
            Rect rect2 = Imgproc.boundingRect(o2);
            return Double.compare(rect1.tl().x, rect2.tl().x);
        });

        for (MatOfPoint contour : contours) {
            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(contour);

            Mat cropped = new Mat(invert, rect);

            boolean turnLeft = rotateToLeft(cropped, 19);
            Mat finalChar;
            if (turnLeft) {
                finalChar = turnLeft(cropped);
            } else {
                finalChar = turnRight(cropped);
            }
            bufferedImage = matToBufferedImage(addPadding(finalChar, 15));
            bufferedImageList.add(bufferedImage);
        }
        return bufferedImageList;
    }

    private static Mat turnLeft(Mat cropped) {
        double previousRotatedWidth = 0;
        double rotatedWidth = 0;
        Mat finalChar = null;
        int i = 0;
        do {
            Mat paddedMat = addPadding(cropped, 40);
            Mat rotateMat = rotateMat(paddedMat, i);

            Mat canny2 = new Mat();
            Canny(rotateMat, canny2, 170, 250);
            List<MatOfPoint> contours1 = findContours(canny2);

            MatOfPoint matOfPoint = contours1.get(0);
            Rect newRect = Imgproc.boundingRect(matOfPoint);
            finalChar = new Mat(rotateMat, newRect);

            if (i == 0) {
                rotatedWidth = finalChar.size().width;
            }

            previousRotatedWidth = rotatedWidth;
            rotatedWidth = finalChar.size().width;
            i += 6;
        } while (rotatedWidth <= previousRotatedWidth);
        return finalChar;
    }

    private static Mat turnRight(Mat cropped) {
        double previousRotatedWidth = 0;
        double rotatedWidth = 0;
        Mat finalChar;
        int i = 0;
        do {
            Mat paddedMat = addPadding(cropped, 40);
            Mat rotateMat = rotateMat(paddedMat, i);

            Mat canny2 = new Mat();
            Canny(rotateMat, canny2, 170, 250);
            List<MatOfPoint> contours1 = findContours(canny2);

            MatOfPoint matOfPoint = contours1.get(0);
            Rect newRect = Imgproc.boundingRect(matOfPoint);
            finalChar = new Mat(rotateMat, newRect);

            if (i == 0) {
                rotatedWidth = finalChar.size().width;
            }

            previousRotatedWidth = rotatedWidth;
            rotatedWidth = finalChar.size().width;
            i -= 6;
        } while (rotatedWidth <= previousRotatedWidth);

        return finalChar;
    }

    private static boolean rotateToLeft(Mat cropped, int angle) {
        Mat paddedMat = addPadding(cropped, 40);
        Mat rotateMat = rotateMat(paddedMat, angle);

        Mat canny2 = new Mat();
        Canny(rotateMat, canny2, 170, 250);
        List<MatOfPoint> contours1 = findContours(canny2);

        MatOfPoint matOfPoint = contours1.get(0);
        Rect newRect = Imgproc.boundingRect(matOfPoint);
        Mat newCropped = new Mat(rotateMat, newRect);

        double defaultWidth = cropped.size().width;
        double rotatedWidth = newCropped.size().width;
        return rotatedWidth < defaultWidth;
    }

    private static Mat rotateMat(Mat mat, int angle) {
        Point center = new Point(mat.width() / 2, mat.height() / 2);
        Mat rotationMatrix2D = getRotationMatrix2D(center, angle, 1);
        Mat dst = new Mat();
        warpAffine(mat, dst, rotationMatrix2D, mat.size(), INTER_AREA, Core.BORDER_DEFAULT, new Scalar(255, 255, 255));
        return dst;
    }

    private static Mat addPadding(Mat cropped, int i) {
        Mat paddedMat = new Mat();
        Core.copyMakeBorder(cropped, paddedMat, i, i, i, i, Core.BORDER_ISOLATED, new Scalar(255, 255, 255));
        return paddedMat;
    }

    private static List<MatOfPoint> findContours(Mat canny) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(canny, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        return contours;
    }

    public static String getCode(String destImgPath) {
        try {
            File sourceFile = new File(destImgPath);
            Tesseract tesseract = getTesseract();
            return tesseract.doOCR(sourceFile);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String getCode(BufferedImage bufferedImage) {
        try {
            Tesseract tesseract = getTesseract();
            return tesseract.doOCR(bufferedImage);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String getCode(List<BufferedImage> bufferedImageList) {
        StringBuilder code = new StringBuilder();
        for (BufferedImage bufferedImage : bufferedImageList) {
            code.append(getCode(bufferedImage));
        }
        return code.toString().replaceAll("\n", "");
    }

    private static Tesseract getTesseract() {
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath("/usr/share/tesseract-ocr/4.00/tessdata/");
        tesseract.setTessVariable("user_defined_dpi", "2400");
        tesseract.setTessVariable("tessedit_char_whitelist", "0123456789");
        return tesseract;
    }
}
