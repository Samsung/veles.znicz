/*
 * Captures the specified region of the screen periodically into bmp files.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
*/
#include <windows.h>
#include <stdio.h>

void SaveBITMAPToFile(const char *szFile, BITMAPINFO *bmi, RGBQUAD *colors)
{
  static BYTE bmp_header[14] = {0x42, 0x4D, 0x36, 0x00, 0x0C, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00};

  HANDLE fout = CreateFile(szFile, GENERIC_WRITE, FILE_SHARE_READ,
               NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  DWORD nn;
  WriteFile(fout, bmp_header, 14, &nn, NULL);

  WriteFile(fout, &bmi->bmiHeader, sizeof(BITMAPINFOHEADER), &nn, NULL);

  WriteFile(fout, colors, bmi->bmiHeader.biWidth * bmi->bmiHeader.biHeight * 4, &nn, NULL);

  CloseHandle(fout);
}

int main(int argc, char **argv)
{
  if(argc < 8)
  {
    printf("USAGE: %s X Y W H TIMEOUT_MS DEST_DIR INITIAL_DELAY_MS\n", argv[0]);
    ExitProcess(1);
  }
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  int w = atoi(argv[3]);
  int h = atoi(argv[4]);
  int timeout = atoi(argv[5]);
  const char *dest_dir = argv[6];
  int initial_delay = atoi(argv[7]);

  HDC dc = CreateCompatibleDC(NULL);

  BITMAPINFO bmi;
  memset(&bmi, 0, sizeof(BITMAPINFO));
  bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmi.bmiHeader.biWidth = w;
  bmi.bmiHeader.biHeight = h;
  bmi.bmiHeader.biPlanes = 1;
  bmi.bmiHeader.biBitCount = 32;
  bmi.bmiHeader.biCompression = BI_RGB;

  RGBQUAD *colors;
  HBITMAP bm = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, (void**)&colors, NULL, 0);

  SelectObject(dc, bm);

  //HFONT font = CreateFont(-MulDiv(40, GetDeviceCaps(dc, LOGPIXELSY), 72), 0,
  //              0, 0, FW_NORMAL, FALSE, FALSE, FALSE, RUSSIAN_CHARSET,
  //              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
  //              DEFAULT_QUALITY, DEFAULT_PITCH, "Times New Roman");

  //SelectObject(dc, font);

  //SetTextColor(dc, RGB(0, 0, 0));
  //SetBkColor(dc, RGB(255, 255, 255));

  //RECT rc = {0, 0, w, h};
  //FillRect(dc, &rc, (HBRUSH)GetStockObject(WHITE_BRUSH));

  //TextOut(dc, 20, 20, "Hello!", 6);

  HDC sDC = GetDC(NULL);

  printf("Will sleep %d msec before capturing\n", initial_delay);
  Sleep(initial_delay);

  for(int frame = 1; ; frame++)
  {
    BitBlt(dc, 0, 0, w, h, sDC, x, y, SRCCOPY);
    GdiFlush();
    SYSTEMTIME tm;
    GetLocalTime(&tm);
    char fnme[512];
    snprintf(fnme, sizeof(fnme), "%s/%04d%02d%02d.%02d%02d%02d.%03d.bmp", dest_dir, (int)tm.wYear, (int)tm.wMonth, (int)tm.wDay,
             (int)tm.wHour, (int)tm.wMinute, (int)tm.wSecond, (int)tm.wMilliseconds);
    SaveBITMAPToFile(fnme, &bmi, colors);
    printf("Captured frame %d to %s\n", frame, fnme);
    Sleep(timeout);
  }

  ReleaseDC(NULL, dc);

  //DeleteObject(font);
  DeleteObject(bm);

  DeleteDC(dc);

  printf("End of job\n");
  ExitProcess(0);
}
