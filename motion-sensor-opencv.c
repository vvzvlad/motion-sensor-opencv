#define _GNU_SOURCE

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>

#define FTDI_OUTPUT 0
int usbdev = -1; /* handle to FTDI device */

#define resolution_x 1280/2 ///1.5
#define resolution_y 960/2 ///1.5
#define LITTLE_MOTION_POINTS_PERCENT 0.5

int dig_key=0;

#define REGIONS_CNT 10
int region_coordinates[REGIONS_CNT][4];
const double MHI_DURATION = 0.5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;

const int N = 2; // cyclic frame buffer
IplImage **buf = 0;
int last = 0;

IplImage *mhi = 0;
IplImage *orient = 0; // orientation
IplImage *mask = 0; // valid orientation mask
IplImage *segmask = 0; // motion segmentation map
CvMemStorage* storage = 0; // temporary storage

/*
 * Signal handlers and data
 */
static unsigned long frames_processed = 0;
static int run = 1;

static void signal_handler(int sig) {
  switch (sig) {
    case SIGINT:
      run = 0;
      printf("Stop processing.\n");
      break;
    case SIGUSR1:
      printf("Statistics:\n"\
             "-frames processed: %lu", frames_processed);
      break;
    default:
      fprintf(stderr, "Unknown signal for processing has been received.");
      break;
  }
}

static void sigusr1_handler(int sig) {
  signal_handler(sig);
}

static void sigint_handler(int sig) {
  signal_handler(sig);
}

/*
 * Main processing functions (computes movement history and detects movement)
 */
static void  update_mhi( IplImage* img, IplImage* dst, int diff_threshold )
{
  double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
  CvSize size = cvSize(img->width,img->height); // get current frame size

  frames_processed++;

  /*
   * Allocate images at the beginning or
   * reallocate them if frame size has changed
   */
  if( !mhi || mhi->width != size.width || mhi->height != size.height ) {
    if( buf == 0 ) {
      buf = (IplImage**)malloc(N*sizeof(buf[0]));
      memset( buf, 0, N*sizeof(buf[0]));
    }

    for( int i = 0; i < N; i++ ) {
      cvReleaseImage( &buf[i] );
      buf[i] = cvCreateImage( size, IPL_DEPTH_8U, 1 );
      cvZero( buf[i] );
    }
    cvReleaseImage( &mhi );
    cvReleaseImage( &orient );
    cvReleaseImage( &segmask );
    cvReleaseImage( &mask );

    mhi = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    cvZero( mhi );                           // clear MHI at the beginning
    orient = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    segmask = cvCreateImage( size, IPL_DEPTH_32F, 1 );
    mask = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  }

  cvCvtColor( img, buf[last], CV_BGR2GRAY ); // convert frame to grayscale

  int idx1 = last,
      idx2 = last = (last + 1) % N;          // index of (last - (N-1))th frame

  IplImage *silh = buf[idx2];
  cvAbsDiff( buf[idx1], buf[idx2], silh );  // get difference between frames

  cvThreshold( silh, silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
  cvUpdateMotionHistory( silh, mhi, timestamp, MHI_DURATION );    // update MHI

  /*
   * convert MHI to green 8u image
   */
  cvCvtScale( mhi, mask, 255./MHI_DURATION,
  (MHI_DURATION - timestamp)*255./MHI_DURATION );
  cvZero( dst );
  cvMerge( 0, mask, 0, 0, dst );

  /*
   * calculate motion gradient orientation and valid orientation mask
   */
  cvCalcMotionGradient( mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );

  if( !storage ) storage = cvCreateMemStorage(0);
  else cvClearMemStorage(storage);

  /*
   * segment motion: get sequence of motion components
   * segmask is marked motion components map. It is not used further
   */
  CvSeq *seq = cvSegmentMotion( mhi, segmask, storage, timestamp, MAX_TIME_DELTA );
  CvRect comp_rect;

  /*
   * iterate through the motion components,
   * One more iteration (i == -1) corresponds to the whole image (global motion)
   */
  for( int i = -1; i < seq->total; i++ ) {

    if( i < 0 ) { // case of the whole image
      comp_rect = cvRect( 0, 0, size.width, size.height );
    } else {      // i-th motion component
      comp_rect = ((CvConnectedComp*)cvGetSeqElem( seq, i ))->rect;
      if( comp_rect.width + comp_rect.height < 25 ) // reject very small components
        continue;
    }

    /*
     * selection of component ROI (Region Of Interest) && counting number of points && reset ROI
     */
    cvSetImageROI( silh,   comp_rect );
    cvSetImageROI( mhi,    comp_rect );
    cvSetImageROI( orient, comp_rect );
    cvSetImageROI( mask,   comp_rect );

    double count = cvNorm( silh, 0, CV_L1, 0 ); // calculate number of points within silhouette ROI

    cvResetImageROI( mhi    );
    cvResetImageROI( orient );
    cvResetImageROI( mask   );
    cvResetImageROI( silh   );

    if ( count < comp_rect.width*comp_rect.height * LITTLE_MOTION_POINTS_PERCENT / 100 ) continue; // check for the case of little motion (too few points has moved)

    /*
     * Number of points is huge, take note this motion && reflect it at user output frame
     */
    if ( comp_rect.width != resolution_x && comp_rect.height != resolution_y )
    {
      cvRectangle(dst,
                  cvPoint(comp_rect.x + comp_rect.width/2+5,comp_rect.y + comp_rect.height/2+5),
                  cvPoint(comp_rect.x + comp_rect.width/2-5,comp_rect.y + comp_rect.height/2-5),
                  CV_RGB(255,0,0), 2, CV_AA, 0 );

      for (int i_mass = 0; i_mass < REGIONS_CNT; i_mass++)
      {
        if (   comp_rect.x + comp_rect.width/2 <=  region_coordinates[i_mass][2]
            && comp_rect.x + comp_rect.width/2 >=  region_coordinates[i_mass][0]
            && comp_rect.y + comp_rect.height/2 <= region_coordinates[i_mass][3]
            && comp_rect.y + comp_rect.height/2 >= region_coordinates[i_mass][1] )
        {
          cvRectangle(dst, 
                      cvPoint(region_coordinates[i_mass][0], region_coordinates[i_mass][1]),
                      cvPoint(region_coordinates[i_mass][2], region_coordinates[i_mass][3]),
                      CV_RGB(0,0,255), 2, CV_AA, 0 );

          printf("Detect motion in region %d\n",i_mass);

#if FTDI_OUTPUT == 1
          char command[strlen("region\n") + (int)log10(REGIONS_CNT) + 1];
          sprintf(command, "region:%d\n", i_mass);
          write(usbdev, command, 2);
#endif
        }
      }
    }
  }
}

void myMouseCallback( int event, int x, int y, int flags, void* param)
{
  (void)flags;
  (void)param;

  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    //printf("%d x %d\n", x, y);
    break;

  case CV_EVENT_LBUTTONDOWN:
    //printf("%d x %d\n", region_coordinates[dig_key][0], region_coordinates[dig_key][1]);
    if (   region_coordinates[dig_key][0] != 0 
        && region_coordinates[dig_key][1] != 0
        && region_coordinates[dig_key][2] == 0
        && region_coordinates[dig_key][3] == 0 )
    {
      region_coordinates[dig_key][2]=x;
      region_coordinates[dig_key][3]=y;
    }
    if (region_coordinates[dig_key][0] == 0 && region_coordinates[dig_key][1] == 0)
    {
      region_coordinates[dig_key][0]=x;
      region_coordinates[dig_key][1]=y;
    }
    break;

  case CV_EVENT_RBUTTONDOWN:
    break;
  case CV_EVENT_LBUTTONUP:
    break;
  }
}

void usage(char *argv0) {
	printf("Usage: %s [-s]\n"\
			" -s :Setup movement detection regions before starting motion detection loop\n", argv0);
}

int main(int argc, char** argv)
{
  (void)argv;
  IplImage* motion = 0;
  CvCapture* capture = 0;
  struct timeval tv0;
  int fps=0;
  int fps_sec=0;
  int now_sec=0;
  char fps_text[255] = {0};

	/*
	 * Check if setup regions is chosen
	 */
	int need_setup_regions = 0;

	switch (argc) {
		case 1:
			printf("Starting movement detection without regions setup.\n");
			need_setup_regions = 0;
			break;
		case 2:
			if (strcmp(argv[1],"-s") == 0) {
				printf("Starting regions setup before movement detection.\n");
				need_setup_regions = 1;
			} else {
				fprintf(stderr, "Bad argument.");
				usage(argv[0]);
				exit(1);
			}
			break;
		default:
			usage(argv[0]);
			exit(2);
	}

  /*
   * setup signal actions (SIGINT - termination, SIGUSR1 - statistics, other - ignore)
   */
  struct sigaction sigact;
  sigact.sa_flags = 0;
  sigfillset(&sigact.sa_mask);
  sigdelset(&sigact.sa_mask, SIGINT);
  sigdelset(&sigact.sa_mask, SIGUSR1);
  pthread_sigmask(SIG_BLOCK, &sigact.sa_mask, NULL);

  sigact.sa_handler = sigusr1_handler;
  sigaction(SIGUSR1, &sigact, NULL);

  sigact.sa_handler = sigint_handler;
  sigaction(SIGINT, &sigact, NULL);

  /*
   * FTDI output setup
   */
#if FTDI_OUTPUT == 1
  system("stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parity -icanon min 1 time 1");
  if ( 0 > usbdev = open("/dev/ttyUSB0", O_RDWR)) {
    fprintf(stderr, "Cannot open FTDI device.\n");
  }
#endif

  /*
   * Use default capture device (its supposed that there is only one cam in the system)
   */
  capture = cvCaptureFromCAM(0);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, resolution_x);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, resolution_y);

  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1.0, 1,1,8);

  if( capture )
  {
    cvNamedWindow( "Motion", 1 );
    FILE* fd = fopen("regions.bin", "rb");  //ключ должен быть "rb" - чтение бинарных данных
    if (fd == NULL)
    {
      printf("Error opening file for reading\n");
      FILE* fd = fopen("regions.bin", "wb");
      if (fd == NULL)
      {
        printf("Error opening file for writing\n");
      }
      else
      {
        fwrite(region_coordinates, 1, sizeof(region_coordinates), fd);
        fclose(fd);
        printf("File created, please restart program\n");
      }
      return 0;
    }

    size_t result = fread(region_coordinates, 1, sizeof(region_coordinates), fd);
    if (result != sizeof(region_coordinates))
      printf("Error size file\n"); //прочитали количество байт не равное размеру массива
    fclose(fd);

    for (int i_regions = 0; i_regions < REGIONS_CNT; i_regions++)
    {
      printf("%d %d %d %d\n", 
             region_coordinates[i_regions][0],
             region_coordinates[i_regions][1],
             region_coordinates[i_regions][2],
             region_coordinates[i_regions][3]);
    }

    if(need_setup_regions) {

      printf("Starting regions setup.\n");
      for(run = 1; run;)
      {
        IplImage* image = cvQueryFrame( capture );
        cvFlip(image, image, 1);
        cvSetMouseCallback( "Motion", myMouseCallback, (void*) image);
        if (   region_coordinates[dig_key][0] != 0 
            && region_coordinates[dig_key][1] != 0 
            && region_coordinates[dig_key][2] == 0 
            && region_coordinates[dig_key][3] == 0 )
        {
          cvRectangle(image, 
              cvPoint(region_coordinates[dig_key][0],  region_coordinates[dig_key][1]),
              cvPoint(region_coordinates[dig_key][0]+1,region_coordinates[dig_key][1]+1),
              CV_RGB(0,0,255), 2, CV_AA, 0 );
        }

        if (   region_coordinates[dig_key][0] != 0 
            && region_coordinates[dig_key][1] != 0
            && region_coordinates[dig_key][2] != 0
            && region_coordinates[dig_key][3] != 0 )
        {  
          cvRectangle(image, 
              cvPoint(region_coordinates[dig_key][0],region_coordinates[dig_key][1]),
              cvPoint(region_coordinates[dig_key][2],region_coordinates[dig_key][3]),
              CV_RGB(0,0,255), 2, CV_AA, 0 );
        }
        cvShowImage( "Motion", image );

        char c = cvWaitKey(20);
        if (c <=57 && c>= 48)
        {
          dig_key=c-48; //key "0123456789"
        }
        if (c == 113 || c == 81) return 0;  //key "q"
        if (c == 99 || c == 67) //key "c"
        {
          region_coordinates[dig_key][0]=0;
          region_coordinates[dig_key][1]=0;
          region_coordinates[dig_key][2]=0;
          region_coordinates[dig_key][3]=0;
        }

        if (c == 120 || c == 88) break; //key "x"
        if (c == 100 || c == 68) //key "d"
        {
          printf("Region cordinates(key): %d %d %d %d\n",
              region_coordinates[dig_key][0],
              region_coordinates[dig_key][1],
              region_coordinates[dig_key][2],
              region_coordinates[dig_key][3]);

          printf("Key: %d/%d\n", dig_key, c);
        }


        if (c >= 0) { //all key
          FILE* fd = fopen("regions.bin", "wb");
          if (fd == NULL)
            printf("Error opening file for writing\n");
          fwrite(region_coordinates, 1, sizeof(region_coordinates), fd);
          fclose(fd);
        }
      }
    }

    printf("Starting motion detection.\n");
    for(run = 1;run;)
    {
      /*
       * Capture frame && alloc mem for output frame if needed
       */
      IplImage* image = cvQueryFrame( capture );

      if( !image )
        break;

      cvFlip(image, image, 1);

      if( !motion )
      {
        motion = cvCreateImage( cvSize(image->width,image->height), 8, 3 );
        cvZero( motion );
        motion->origin = image->origin;
      }

      /*
       * Compute output frame
       */
      update_mhi( image, motion, 30 );

      /*
       * FPS computation && printing to output frame
       */
      gettimeofday(&tv0,0);
      now_sec=tv0.tv_sec;
      if (fps_sec == now_sec)
      {
        fps++;
      }
      else
      {
        fps_sec=now_sec;
        snprintf(fps_text,254,"%d",fps);
        fps=0;
      }
      cvPutText(motion, fps_text, cvPoint(5, 20), &font, CV_RGB(255,255,255));
      cvShowImage( "Motion", motion );

      /*
       * Needed to switch context and allow output frame to be shown to user
       */
      cvWaitKey(10);
    }

    cvReleaseCapture( &capture );
    cvReleaseImage(&motion);
    cvDestroyWindow( "Motion" );
  } else {
		fprintf(stderr, "Cannot open default camera! Terminating.\n");
	}

  return 0;
  fcloseall();
#if FTDI_OUTPUT == 1
  close(usbdev);
#endif
}

