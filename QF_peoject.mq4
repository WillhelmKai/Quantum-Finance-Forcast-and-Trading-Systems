//+------------------------------------------------------------------+
//|                                                      project.mq4 |
//|                                                   Dr.Raymond Lee |
//|                                                         qffc.org |
//+------------------------------------------------------------------+
#property copyright "Dr.Raymond Lee"
#property link      "qffc.org"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
//int n = 3600 // past time period
//int t = 30  //prediction time in n+t
string      QP_Directory   = "Project_Data";       // data Directory

string  TP_Code[8]={ "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF", "EURAUD","GBPJPY", "USDJPY"};
int maxTP = 8; // the num of product
int maxTS = 200;//2048; //max no of Time Series Record

int read_data(string TPSymbol){
      int        DT_YY[2048];
      int        DT_MM[2048];
      int        DT_DD[2048];
      double     DT_OP[2048];
      double     DT_HI[2048];
      double     DT_LO[2048];
      double     DT_CL[2048];
      double     DT_VL[2048];
      double     DT_RT[2048];
      double     DT_EL[2048];
      
      string Data_FileName = TPSymbol+".csv";
      
      FileDelete(QP_Directory+"//"+Data_FileName,FILE_COMMON);
      ResetLastError();
      int QPD_FileHandle    = FileOpen(QP_Directory+"//"+Data_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

      // Write Header Line
      FileWrite(QPD_FileHandle,"Year","Month","Day","Open","High","Low","Close","Volumn","Return","QantumPriceLevel");
  

      //check TSsize we can retrive
      Print("product ",TPSymbol);
      int TSsize = 0;
      while (iTime(TPSymbol,PERIOD_D1,TSsize)>0 && (TSsize<maxTS))
      {
         TSsize++;
      }
         
      if(TSsize == 0) {
      Print("no data for ",TPSymbol);
      return -1;//no data for this product
      }
      
      for (int d=1;d<TSsize;d++)
      {
          DT_YY[d-1] = TimeYear(iTime(TPSymbol,PERIOD_D1,d));     
          DT_MM[d-1] = TimeMonth(iTime(TPSymbol,PERIOD_D1,d));     
          DT_DD[d-1] = TimeDay(iTime(TPSymbol,PERIOD_D1,d));     
          DT_OP[d-1] = iOpen(TPSymbol,PERIOD_D1,d);
          DT_HI[d-1] = iHigh(TPSymbol,PERIOD_D1,d);
          DT_LO[d-1] = iLow(TPSymbol,PERIOD_D1,d);
          DT_CL[d-1] = iClose(TPSymbol,PERIOD_D1,d);
          DT_VL[d-1] = iVolume(TPSymbol,PERIOD_D1,d);
          
          //get price return
          if(d >=2 && DT_CL[d-1] >0){
          DT_RT[d-1] = DT_CL[d-1]/DT_CL[d-2] ; 
          }else{
          DT_RT[d-1] = 1;          
          }
          
         Print(TPSymbol,"  ",DT_YY[d-1],"  ",DT_MM[d-1],"  ",DT_DD[d-1],"  OP:",DT_OP[d-1],
         "HI:",DT_HI[d-1],"  LO:",DT_LO[d-1],"  CL:",DT_CL[d-1],"  VL:",DT_VL[d-1],"  RT:",DT_RT[d-1]"  EL: TBD");
         
         
         FileWrite(QPD_FileHandle,DT_YY[d-1],DT_MM[d-1],DT_DD[d-1],
             DoubleToString(DT_OP[d-1],8),DoubleToString(DT_HI[d-1],8),
             DoubleToString(DT_LO[d-1],8),DoubleToString(DT_CL[d-1],8),
             DT_VL[d], DoubleToString(DT_RT[d-1],8));
      }     
      FileClose(QPD_FileHandle);

     return 0;
}



int init(){

for(int i = 0; i<maxTP; i++)
   //go through every single product
   {
   read_data(TP_Code[i]);
      
   }


return 0;
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
