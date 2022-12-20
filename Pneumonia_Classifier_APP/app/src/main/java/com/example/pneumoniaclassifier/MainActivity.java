package com.example.pneumoniaclassifier;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.pneumoniaclassifier.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    private static final int FROM_ALBUM = 1;    // onActivityResult 식별자

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar); //앱 상단 툴바
        binding.checkBtn.setVisibility(View.INVISIBLE); //이미지 업로드 전에는 "진단하기" 버튼 사라짐


        //이미지 업로드 버튼 클릭 이벤트
        binding.uploadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // 이미지 파일 선택 팝업창
                Intent intent = new Intent();
                intent.setType("image/*");                      // 이미지만
                intent.setAction(Intent.ACTION_GET_CONTENT);    // 카메라(ACTION_IMAGE_CAPTURE)
                startActivityForResult(intent,FROM_ALBUM);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // 카메라를 다루지 않기 때문에 앨범 상수에 대해서 성공한 경우에 대해서만 처리
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode != FROM_ALBUM || resultCode != RESULT_OK)
            return;

        //각 모델에 따른 input , output shape 각자 맞게 변환
        float[][][][] input = new float[1][150][150][1];
        float[][] output = new float[1][1];
        System.out.println(output);

        try {
            int batchNum = 0;
            InputStream buf = getContentResolver().openInputStream(data.getData());
            Bitmap bitmap = BitmapFactory.decodeStream(buf);
            buf.close();

            //이미지 뷰에 선택한 사진 띄우기
            binding.uploadImage.setScaleType(ImageView.ScaleType.FIT_XY);
            binding.uploadImage.setImageBitmap(bitmap);
            binding.checkBtn.setVisibility(View.VISIBLE);
            binding.textDate.setText("");
            binding.textDiagnosis.setText("");

            // x,y 최댓값 사진 크기에 따라 달라짐 (조절)
            for (int x = 0; x < 150; x++) {
                for (int y = 0; y < 150; y++) {
                    int pixel = bitmap.getPixel(x, y);
                    input[batchNum][x][y][0] = pixel;
                }
            }

            //진단하기 버튼 클릭 이벤트
            binding.checkBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // 자신의 tflite 이름 써주기
                    Interpreter lite = getTfliteInterpreter("myModel.tflite");
                    lite.run(input, output);

                    for (int i = 0; i < 2; i++) {
                            if (output[0][0] == 0.0) {
                                binding.textDiagnosis.setText("음성");
                                binding.textDiagnosis.setTextColor(Color.parseColor("#000000"));
                            } else if (output[0][0] == 1.0) {
                                binding.textDiagnosis.setText("양성");
                                binding.textDiagnosis.setTextColor(Color.parseColor("#FF4400"));
                            }
                    }

                    // 진단 요청한 시간 출력
                    DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    String dateTime = dateFormat.format(new Date());
                    binding.textDate.setText(dateTime);
                }
            });

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(MainActivity.this, modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /*
     * 메뉴 버튼 네비게이션에서 사용하는 메소드 (테스트에서 사용 안해 주석처리)

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }

     */


}