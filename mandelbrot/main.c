 /*
 c program:
 --------------------------------
  1. draws Mandelbrot set for Fc(z)=z*z +c
  using Mandelbrot algorithm ( boolean escape time )
 -------------------------------
 2. technique of creating ppm file is  based on the code of Claudio Rocchini
 http://en.wikipedia.org/wiki/Image:Color_complex_plot.jpg
 create 24 bit color graphic file ,  portable pixmap file = PPM
 see http://en.wikipedia.org/wiki/Portable_pixmap
 to see the file use external application ( graphic viewer)
  */
 #include <stdio.h>
 #include <math.h>
 int main()
 {
   FILE * fp;
   char *filename="new1.ppm";
   char *comment="# ";/* comment should start with # */
   static unsigned char color[3];

   /*create new file,give it a name and open it in binary mode  */
   fp= fopen(filename,"wb"); /* b -  binary mode */
   /*write ASCII header to the file*/
   fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,iXmax,iYmax,MaxColorComponentValue);
   /* compute and write image data bytes to the file*/
   
   void calc_mandel(fp)
   
   fclose(fp);
   return 0;
 }
 
void hsv_to_rgb(int hue, int min, int max, rgb_t *p) {
	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p->r = p->g = p->b = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
	double c = VAL * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p->r = p->g = p->b = 0;
 
	switch((int)h) {
	case 0: p->r = c; p->g = X; return;
	case 1:	p->r = X; p->g = c; return;
	case 2: p->g = c; p->b = X; return;
	case 3: p->g = X; p->b = c; return;
	case 4: p->r = X; p->b = c; return;
	default:p->r = c; p->b = X;
	}
}
 
void calc_mandel(FILE *fp) {
    /* screen ( integer) coordinate */
   int iX,iY;
   const int iXmax = 1200;
   const int iYmax = 1200;
   /* world ( double) coordinate = parameter plane*/
   double Cx,Cy;
   const double CxMin=-2.5;
   const double CxMax=1.5;
   const double CyMin=-2.0;
   const double CyMax=2.0;
   /* */
   double PixelWidth=(CxMax-CxMin)/iXmax;
   double PixelHeight=(CyMax-CyMin)/iYmax;
   /* color component ( R or G or B) is coded from 0 to 255 */
   /* it is 24 bit color RGB file */
   const int MaxColorComponentValue=255;
   
   /* Z=Zx+Zy*i  ;   Z0 = 0 */
   double Zx, Zy;
   double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
   /*  */
   int Iteration;
   const int IterationMax=100000;
   /* bail-out value , radius of circle ;  */
   const double EscapeRadius=5;
   double ER2=EscapeRadius*EscapeRadius;
   
   for(iY=0;iY<iYmax;iY++)
  {
 Cy=CyMin + iY*PixelHeight;
 if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
 for(iX=0;iX<iXmax;iX++)
 {
            Cx=CxMin + iX*PixelWidth;
            /* initial value of orbit = critical point Z= 0 */
            Zx=0.0;
            Zy=0.0;
            Zx2=Zx*Zx;
            Zy2=Zy*Zy;
            /* */
            for (Iteration=0;Iteration<IterationMax && ((Zx2+Zy2)<ER2);Iteration++)
            {
                Zy=2*Zx*Zy + Cy;
                Zx=Zx2-Zy2 +Cx;
                Zx2=Zx*Zx;
                Zy2=Zy*Zy;
            };
            /* compute  pixel color (24 bit = 3 bytes) */
            if (Iteration==IterationMax)
            { /*  interior of Mandelbrot set = black */
               color[0]=0;
               color[1]=0;
               color[2]=0;
            }
         else
            { /* exterior of Mandelbrot set = white */
                 color[0]=255; /* Red*/
                 color[1]=255;  /* Green */
                 color[2]=255;/* Blue */
            };
            /*write color to the file*/
            fwrite(color,1,3,fp);
    }
  }
   
	for (i = 0; i < height; i++) {
		px = tex[i];
		y = (i - height/2) * scale + cy;
		for (j = 0; j  < width; j++, px++) {
			x = (j - width/2) * scale + cx;
			iter = 0;
 
			zx = hypot(x - .25, y);
			if (x < zx - 2 * zx * zx + .25) iter = max_iter;
			if ((x + 1)*(x + 1) + y * y < 1/16) iter = max_iter;
 
			zx = zy = zx2 = zy2 = 0;
			for (; iter < max_iter && zx2 + zy2 < 4; iter++) {
				zy = 2 * zx * zy + y;
				zx = zx2 - zy2 + x;
				zx2 = zx * zx;
				zy2 = zy * zy;
			}
			if (iter < min) min = iter;
			if (iter > max) max = iter;
			*(unsigned short *)px = iter;
		}
	}
 
	for (i = 0; i < height; i++) {
		for (j = 0, px = tex[i]; j  < width; j++, px++) {
			hsv_to_rgb(*(unsigned short*)px, min, max, px);
			fwrite(px,1,3,fp);
		}
	}
	
	
}