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
 #include <stdlib.h>
 
struct {
   int iXmax,iYmax;
   double CxMin,CxMax,CyMin,CyMax;
   int IterationMax;
   int EscapeRadius;
} typedef mandelbrot_t;

struct {
   unsigned char r,g,b;
} typedef rgb_t;
 
rgb_t **tex = 0;

void hsv_to_rgb(int hue, int min, int max, rgb_t *p) {
   int invert, saturation, color_rotate, VAL;
   
   invert = 0;
   saturation = 1;
   color_rotate = 0;
   VAL = 255;
   
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
 
void calc_mandel(mandelbrot_t *m, FILE *fp) {
    /* screen (integer) coordinate */
   int iX,iY;
   const int iXmax = m->iXmax;
   const int iYmax = m->iYmax;
   
   /* world (double) coordinate = parameter plane*/
   double Cx,Cy;
   const double CxMin = m->CxMin;
   const double CxMax = m->CxMax;
   const double CyMin = m->CyMin;
   const double CyMax = m->CyMax;
   /* */
   double PixelWidth=(CxMax-CxMin)/iXmax;
   double PixelHeight=(CyMax-CyMin)/iYmax;
   
   // Used to get even gradient
   int min,max;
   min = max = 0;
   rgb_t *px;

   /* Z=Zx+Zy*i  ;   Z0 = 0 */
   double Zx, Zy;
   double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
   /*  */
   int Iteration;
   const int IterationMax = m->IterationMax;
   /* bail-out value , radius of circle ;  */
   const double EscapeRadius = m->EscapeRadius;
   double ER2=EscapeRadius*EscapeRadius;
   
   for(iY=0;iY<iYmax;iY++) {
      px = tex[iY];
      Cy=CyMin + iY*PixelHeight;
      if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
      for(iX=0;iX<iXmax;iX++) {
         Cx=CxMin + iX*PixelWidth;
         /* initial value of orbit = critical point Z= 0 */
         Zx=0.0;
         Zy=0.0;
         Zx2=Zx*Zx;
         Zy2=Zy*Zy;
         /* */
         for (Iteration=0;Iteration<IterationMax && ((Zx2+Zy2)<ER2);Iteration++) {
            Zy=2*Zx*Zy + Cy;
            Zx=Zx2-Zy2 +Cx;
            Zx2=Zx*Zx;
            Zy2=Zy*Zy;
         };
         if (Iteration < min) min = Iteration;
         if (Iteration > max) max = Iteration;
         *(unsigned short *)px = Iteration;
      }
   }
  
   int i, j;
 
	for (i = 0; i < iXmax; i++) {
		for (j = 0, px = tex[i]; j  < iYmax; j++, px++) {
			hsv_to_rgb(*(unsigned short*)px, min, max, px);
			fwrite(px,1,3,fp);
		}
	}
	
	
}

int main() {
   FILE * fp;
   char *filename="new1.ppm";
   char *comment="# ";/* comment should start with # */
   
   mandelbrot_t m;
   
   m.iXmax = 1200;
   m.iYmax = 1200;
   
   m.CxMin = -2.5;
   m.CxMax = 1.5;
   
   m.CyMin = -2.0;
   m.CyMax = 2.0;
   
   m.IterationMax=100000;
   m.EscapeRadius=5;
   
   /* color component ( R or G or B) is coded from 0 to 255 */
   /* it is 24 bit color RGB file */
   const int MaxColorComponentValue=255;
   
   tex = malloc(sizeof(rgb_t *) * m.iXmax);
   tex[0] = malloc(sizeof(rgb_t) * m.iXmax * m.iYmax);

   int i;

   for(i = 0; i < m.iXmax; i++) 
        tex[i] = (*tex + m.iYmax * i);
   
   /*create new file,give it a name and open it in binary mode  */
   fp= fopen(filename,"wb"); /* b -  binary mode */
   
   /*write ASCII header to the file*/
   fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,m.iXmax,m.iYmax,MaxColorComponentValue);
   /* compute and write image data bytes to the file*/
   
   calc_mandel(&m, fp);
   
   fclose(fp);
   return 0;
}
