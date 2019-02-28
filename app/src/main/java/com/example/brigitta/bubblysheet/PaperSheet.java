package com.example.brigitta.bubblysheet;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;
import static org.opencv.imgproc.Imgproc.getStructuringElement;


public class PaperSheet extends AppCompatActivity {

    private String[] options = new String[]{"A", "B", "C", "D", "E"};
    private int questionCount = 30;
    private List<Integer> answers;
    private List<Integer> correctAnswers;
    private List<MatOfPoint> bubbles;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFormat(PixelFormat.RGBA_8888);
        setContentView(R.layout.activity_paper_sheet);

        long PaperCopy = getIntent().getLongExtra("PaperSheet", 0);
        Mat TempPaperCopy = new Mat (PaperCopy);
        Mat PaperSheet = TempPaperCopy.clone();

        Mat mErode = new Mat();
        Mat mGray = new Mat();
        Mat mBlur = new Mat();
        Mat mThresh = new Mat();
        Mat mAdaptiveThresh = new Mat();

        Imgproc.erode(PaperSheet, mErode, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1)));
        Imgproc.cvtColor(mErode, mGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(mGray, mBlur, new Size(5, 5));
        Imgproc.threshold(mGray, mThresh, 140, 255, THRESH_BINARY);
        Imgproc.adaptiveThreshold(mBlur, mAdaptiveThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

        //this.questionCount = questionCount;
        correctAnswers = new ArrayList<>();
        answers = new ArrayList<>();
        bubbles = new ArrayList<>();

        findBubbles(PaperSheet, mAdaptiveThresh);
        findAnswers(PaperSheet, mThresh);
        setCorrectAnswers();


        for (int i = 0; i < answers.size(); i++) {
            Integer optionIndex = answers.get(i);
            System.out.println ((i + 1) + ". " + (optionIndex == null ? "EMPTY/INVALID" : options[optionIndex]));
            System.out.println((i + 1) + "a: " + options[correctAnswers.get(i)]);
        }

        if (correctAnswers.size() != answers.size()) {
            Toast.makeText(getApplicationContext(), "Number of problem doesn't match.", Toast.LENGTH_SHORT).show();
            //error
        }

        int correct = 0;
        for (int j = 0; j < correctAnswers.size(); j++) {
            if (answers.get(j) == correctAnswers.get(j)) {
                correct ++;
            }
        }
        System.out.println("Correct problems: " + correct);
        System.out.println("Total problems: " + correctAnswers.size());

        System.out.println("Score: " + (correct * 10) / 3);

        Imgproc.cvtColor(PaperSheet, PaperSheet, Imgproc.COLOR_BGR2RGB);

        Bitmap Paper = Bitmap.createBitmap(PaperSheet.cols(), PaperSheet.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(PaperSheet, Paper);

        ImageView iv = (ImageView) findViewById(R.id.imageView1);
        iv.setImageBitmap(Paper);

        mErode.release();
        mGray.release();
        mBlur.release();
        mThresh.release();
        mAdaptiveThresh.release();
    }

    private void findBubbles(Mat PaperSheet, Mat mAdaptiveThresh) {

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mAdaptiveThresh, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        List<MatOfPoint> drafts = new ArrayList<>();
        //To determine which regions of the image are bubbles, we first loop over each of the individual contours.
        for (MatOfPoint contour : contours) {
            //Compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
            Rect box = Imgproc.boundingRect(contour);
            int w = box.width;
            int h = box.height;
            double ratio = Math.max(w, h) / Math.min(w, h);

            //In order for a contour area to be considered a bubble, the region should:
            //1. Be sufficiently wide and tall (in this case, at least 20-30 pixels in both dimensions).
            //2. Have an aspect ratio that is approximately equal to 1.
            //As long as these checks hold, we can update our drafts list and mark the region as a bubble.

            if (ratio >= 0.9 && ratio <= 1.1) {
                if (Math.max(w, h) < 30 && Math.min(w, h) > 20) {
                    drafts.add(contour);
                    //Imgproc.drawContours(PaperSheet, contours, contours.indexOf(contour), new Scalar(0, 0, 255), 1);
                }
            }
        }

        if(drafts.size() != questionCount * options.length){
            Toast.makeText(getApplicationContext(), "Couldn't capture all bubbles. Try again. ", Toast.LENGTH_SHORT).show();
        }

        //First, we must sort our drafts from top-to-bottom.
        // This will ensure that rows of questions that are closer to the top of the exam will appear first in the sorted list.
        sortTopLeft2BottomRight(drafts);

        bubbles = new ArrayList<>();

        //The reason this methodology works is because we have already sorted our contours from top-to-bottom.
        //We know that the 5 bubbles for each question will appear sequentially in our list — but we do not know whether these bubbles will be sorted from left-to-right.
        //Each question has 5 possible answers.
        //Sort the current set of contours from left to right.
        for (int j = 0; j < drafts.size(); j += options.length * 2) {
            List<MatOfPoint> row = drafts.subList(j, j + options.length * 2);
            sortLeft2Right(row);
            bubbles.addAll(row);
        }
    }

    //Given a row of bubbles, the next step is to determine which bubble is filled in.
    //We can accomplish this by using our thresh image and counting the number of non-zero pixels (i.e., foreground pixels) in each bubble region:

    private void findAnswers(Mat PaperSheet, Mat mThresh) {

        for (int i = 0; i < bubbles.size(); i += options.length) {
            List<MatOfPoint> rows = bubbles.subList(i, i + options.length);

            int[][] filled = new int[rows.size()][5];

            //Loop over the sorted contours
            for (int j = 0; j < rows.size(); j++) {
                MatOfPoint col = rows.get(j);
                List<MatOfPoint> list = Arrays.asList(col);

                //Construct a mask that reveals only the current "bubble" for the question
                Mat mask = new Mat(mThresh.size(), CvType.CV_8UC1);
                Imgproc.drawContours(mask, list, -1, new Scalar(255, 0, 0), -1);

                //Apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area.
                Mat conjuction = new Mat(mThresh.size(), CvType.CV_8UC1);
                Core.bitwise_and(mThresh, mask, conjuction); //countNonZero counts black pixel in the mask

                int countNonZero = Core.countNonZero(conjuction);
                filled[j] = new int[]{countNonZero, i, j};
            }

            int[] selection = chooseFilledCircle(filled);

            if (selection != null) {
                Imgproc.drawContours(PaperSheet, Arrays.asList(rows.get(selection[2])), -1, new Scalar(0, 255, 0), 2);
            }

            answers.add(selection == null ? null : selection[2]);

        }

        List<Integer> odds = new ArrayList<>();
        List<Integer> evens = new ArrayList<>();
        for(int i = 0; i < answers.size(); i++) {
            if(i % 2 == 0) odds.add(answers.get(i));
            if(i % 2 == 1) evens.add(answers.get(i));
        }

        answers.clear();
        answers.addAll(odds);
        answers.addAll(evens);
    }

    private void setCorrectAnswers() {
        correctAnswers = new ArrayList<>();
        correctAnswers.add(1); //1
        correctAnswers.add(1); //2
        correctAnswers.add(1); //3
        correctAnswers.add(2); //4
        correctAnswers.add(2); //5
        correctAnswers.add(3); //6
        correctAnswers.add(1); //7
        correctAnswers.add(4); //8
        correctAnswers.add(4); //9
        correctAnswers.add(0); //10
        correctAnswers.add(3); //11
        correctAnswers.add(1); //12
        correctAnswers.add(3); //13
        correctAnswers.add(1); //14
        correctAnswers.add(4); //15
        correctAnswers.add(1); //16
        correctAnswers.add(2); //17
        correctAnswers.add(1); //18
        correctAnswers.add(2); //19
        correctAnswers.add(2); //20
        correctAnswers.add(3); //21
        correctAnswers.add(4); //22
        correctAnswers.add(3); //23
        correctAnswers.add(4); //24
        correctAnswers.add(0); //25
        correctAnswers.add(1); //26
        correctAnswers.add(2); //27
        correctAnswers.add(3); //28
        correctAnswers.add(1); //29
        correctAnswers.add(4); //30
    }

    private int[] chooseFilledCircle(int[][] rows){

        double mean = 0;
        for(int i = 0; i < rows.length; i++){
            mean += rows[i][0];
        }
        mean = 1.0d * mean / options.length;

        int anomalouses = 0;
        for(int i = 0; i < rows.length; i++){
            if(rows[i][0] > mean) anomalouses++;
        }

        if(anomalouses == options.length - 1){
            int[] lower = null;
            for(int i = 0; i < rows.length; i++){
                if(lower == null || lower[0] > rows[i][0]){
                    lower = rows[i];
                }
            }

            return lower;

        } else {
            return null;
        }
    }

    private static void sortTopLeft2BottomRight(List<MatOfPoint> drafts) {
        // top-left to right-bottom sort
        Collections.sort(drafts, (e1, e2) -> {

            Point o1 = new Point(e1.get(0, 0));
            Point o2 = new Point(e2.get(0, 0));

            return o1.y > o2.y ? 1 : -1;
        });
    }

    public static void sortLeft2Right(List<MatOfPoint> row){
        // left to right sort
        Collections.sort(row, (e1, e2) -> {

            Point o1 = new Point(e1.get(0, 0));
            Point o2 = new Point(e2.get(0, 0));

            return o1.x > o2.x ? 1 : -1;
        });
    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, MainActivity.class));
        finish();
    }

}

