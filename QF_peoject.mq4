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
string      Prediction_Directory = "Prediction";//prediction directory

string  TP_Code[8]={ "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCHF", "EURAUD","GBPJPY", "USDJPY"};
int maxTP = 8; // the num of product
int maxTS = 1024;//2048; //max no of Time Series Record



double all_NQPL[8][21]; 

string Predicted_value[8];

int read_data(string TPSymbol, double &return_array[]){
      //declaration varuables
      //basic data
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
      
      
      double     mu = 0;
      double     sigma = 0;
      double     dr = 0;
      
      double     Q[100];//wave function
      double     NQ[100];//normalized wave function
      double     r[100]; //return rate under normal distribution
      
      double     K[21];
      double     QFEL[21];
      double     QPR[21];
      double     NQPR[21];
      
      
      
      //define the destination add
      string Data_FileName = TPSymbol+".csv";
      
      FileDelete(QP_Directory+"//"+Data_FileName,FILE_COMMON);
      ResetLastError();
      int QPD_FileHandle    = FileOpen(QP_Directory+"//"+Data_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');

      // Write Header Line
      FileWrite(QPD_FileHandle,"Year","Month","Day","Open","High","Low","Close","Volumn","Return","QPL1","QPL2","QPL3","QPL4","QPL-1","QPL-2","QPL-3","QPL-4");
  

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
          
         //calculate mu 
         mu = mu+DT_RT[d-1];
         
      }
      mu = mu/TSsize;
      
      // Calculate STDEV sigma
      for (int d=0;d<TSsize;d++)
      {
        sigma = sigma + (DT_RT[d]-mu)*(DT_RT[d]-mu);
      }
      sigma = sqrt((sigma / TSsize));      
      
      
      // Calculate dr where dr = 3*sigma/50
      dr = 3 * sigma / 50;


      //transform into wave function
      int tQno = 0;
      double zero_index = 1 - 50*dr;
      for(int d=0; d< TSsize; d++) //travers the DT_RT
      {
         //retrive the index of Q(r) with the value of DT_RT[d]         
         int index = int((DT_RT[d]-zero_index)/dr);
         //in the position of index accumulate the index
         if(index <0 || index>99)
         {
           continue;
         }
         Q[index] += 1;
         tQno += 1;
      }  
      
      
      double auxR = 1 - (dr * 50);
      for (int nQ=0;nQ<100;nQ++)
      {
         r[nQ]  = auxR;
         NQ[nQ] = Q[nQ]/tQno;               
         auxR = auxR + dr;
      }     
      
      
      //find maxQ and maxQno
      double maxQ   = 0;
      int maxQno = 0;
      for (int nQ=0;nQ<100;nQ++)
      {
         if (NQ[nQ] > maxQ)
         {
            maxQ   = NQ[nQ];
            maxQno = nQ; 
         }       
      }
      
      //find K
      for (int eL=0;eL<21;eL++)
      {
         K[eL] = MathPow((1.1924 + (33.2383*eL) + (56.2169*eL*eL))/(1 + (43.6106 *eL)),(1.0/3.0));
      }
      
      //find lambda
      double r0  = r[maxQno] - (dr/2);
      double r1  = r0 + dr;
      double rn1 = r0 - dr;
      double Lup = (pow(rn1,2)*NQ[maxQno-1])-(pow(r1,2)*NQ[maxQno+1]);
      double Ldw = (pow(rn1,4)*NQ[maxQno-1])-(pow(r1,4)*NQ[maxQno+1]);
      double L   = MathAbs(Lup/Ldw);
      
      //calculate QFEL by using L, K
      for (int eL=0;eL<21;eL++)
      {
         double p = -1 * pow((2*eL+1),2);
         double q = -1 * L * pow((2*eL+1),3) * pow(K[eL],3);
         
         // Apply Cardano's Method to find the real root of the depressed cubic equation
         double u = MathPow((-0.5*q + MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),(1.0/3.0));
         double v = MathPow((-0.5*q - MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),(1.0/3.0));
         
         // Store the QFEL 
         QFEL[eL] = u + v;
         
         
      }
      
      //find out QPR and NQPR
      for (int eL=0;eL<21;eL++)
      {     
         QPR[eL]  = QFEL[eL]/QFEL[0];
         NQPR[eL] = 1 + 0.21*sigma*QPR[eL];
         return_array[eL] = NQPR[eL];
      }
      
      
      for (int d=1;d<TSsize;d++)
      {          
      //generate energy level and dump into csv
         double QPLn1,QPLn2,QPLn3,QPLn4 ;
         if(NQPR[3] != 0.0) QPLn1 = DT_CL[d-1] / NQPR[3];
         if(NQPR[2] != 0.0) QPLn2 = DT_CL[d-1] / NQPR[2];
         if(NQPR[1] != 0.0) QPLn3 = DT_CL[d-1] / NQPR[1];
         if(NQPR[0] != 0.0) QPLn4 = DT_CL[d-1] / NQPR[0];   
         
            
         double QPL1 = DT_CL[d-1] * NQPR[0];
         double QPL2 = DT_CL[d-1] * NQPR[1];
         double QPL3 = DT_CL[d-1] * NQPR[2];
         double QPL4 = DT_CL[d-1] * NQPR[3]; 
      
      
      
         //Print(TPSymbol,"  ",DT_YY[d-1],"  ",DT_MM[d-1],"  ",DT_DD[d-1],"  OP:",DT_OP[d-1],
         //"HI:",DT_HI[d-1],"  LO:",DT_LO[d-1],"  CL:",DT_CL[d-1],"  VL:",DT_VL[d-1],"  RT:",DT_RT[d-1],
         //"  QPL1: ",QPL1,"  QPL2: ",QPL2,"  QPL3: ",QPL3,"  QPL4: ",QPL4,
         //"  QPLn1: ",QPLn1,"  QPLn2: ",QPLn2,"  QPLn3: ",QPLn3,"  QPLn4: ",QPLn4);
         
         
         FileWrite(QPD_FileHandle,DT_YY[d-1],DT_MM[d-1],DT_DD[d-1],
             DoubleToString(DT_OP[d-1],8),DoubleToString(DT_HI[d-1],8),
             DoubleToString(DT_LO[d-1],8),DoubleToString(DT_CL[d-1],8),
             DT_VL[d], DoubleToString(DT_RT[d-1],8), 
             DoubleToString(QPL1,8),DoubleToString(QPL2,8),DoubleToString(QPL3,8),DoubleToString(QPL4,8),
             DoubleToString(QPLn1,8),DoubleToString(QPLn2,8),DoubleToString(QPLn3,8),DoubleToString(QPLn4,8)
             );
      }     
      FileClose(QPD_FileHandle);

     return 0;
}

double QPL_zeroState;
double QPL_negState[21];
double QPL_posState[21];
//-----------------------------------------------------
//Calculate the QPL zero, positive, and negative states
//-----------------------------------------------------
void calculateQPL_states(double& NQPR[], double open){

   for (int i = 0; i<21; i++) {

                                       // if i between 1 and 20 calculate the positive and negative states
 
          QPL_posState[i] = open * NQPR[i];
          QPL_negState[i] = open/ NQPR[i]; 
          
                
      }

return ;
      }


//-----------------------------------------------------------
//Get the Current Price of the product from the TP_Code array
//-----------------------------------------------------------
double currentPrice(string productSymbol)
{



MqlTick last_tick;
double ask;
   if(SymbolInfoTick(productSymbol,last_tick)) //getting the ask price at the moment and returnign to trade function
     {  
             ask = last_tick.ask; 
             
     } 
   else Print("SymbolInfoTick() failed, error = ",GetLastError());
return ask;                              //returning the current ask price

}

//---------------------------------------------------------
//compare the current price to the QPL for trading strategy
//---------------------------------------------------------
int quantumLevel(double ask){
   
  int energyLevel = 0;
  
                //else if the ask < zero state QPL loop negative QPL
         for (int i = 1;  i<21; i++)
         {
         
            if(ask<QPL_negState[i])
            {
       
               i++;
       
            }
            else
            {
               energyLevel = -i;
            }
            
           }
          
         
         return energyLevel;
  }


double vPoint; 
int vSlippage;   
double prClose ; 
//double ticket[];
void buyFunction(string symbol, double ask, int risk, double prValue){
double volume = .5;

int  Slippage;
 //Detect 3/5 digit brokers for Point and Slippage
if (Point == 0.00001) 
{ vPoint = 0.0001; vSlippage = Slippage *10;}

else {
if (Point == 0.001) 
{ vPoint = 0.01; vSlippage = Slippage *10;}

else vPoint = Point; vSlippage = Slippage;
} 
int ticket;
double sloss = ask *.9995 ;
double sprofit= ask * 1.0003;
 
vSlippage = 10 ;
ResetLastError();
 //if(ask < prValue){
   //if(0<risk<4)
   //{

    //int ticket = OrderSend(symbol, OP_BUY, volume,ask,vSlippage,sloss,sprofit,NULL, 1,0, clrNONE);
     //Print("buying product ",symbol," volume ",volume,"  Slippage ",vSlippage,"  ask ",ask," ticket ",ticket);
     
    // Alert(GetLastError());
    //}
 //else{
   if(1<risk<21)
   {  

   int ticket = OrderSend(symbol, OP_BUY, volume,ask,vSlippage,sloss,sprofit,NULL,1,0, clrNONE);
   Print("buying product ",symbol," volume ",volume,"  Slippage ",vSlippage,"  ask ",ask," ticket ",ticket);
     //Alert(GetLastError());
   }
   else{ Print("No order Executed");}


//}  
return;


}
//extern double  cPRICE = 0;               // Current Price
//extern string  sPRICE="";                // Current Price string

//----------------------------------------
// Main trading program
//----------------------------------------
int quantumTrader(double& NQPL[], string symbol,double prValue){ 

double cPRICE = currentPrice(symbol);                   //Get the current Price of the product
double open = iOpen(symbol,PERIOD_D1, 0);        //Get the daily open to calculate the QPL
//sPRICE = DoubleToString(cPRICE,5);
                                                          
//for (int i=0; i<21; i++){
//NQPL[i] = temp[i];
//}
calculateQPL_states(NQPL,open); //Calculate the QPL zero, pos, neg



int eLevel = quantumLevel(cPRICE);                   //Compare the current price and return the current energy level
int risk = eLevel + 21;    
buyFunction(symbol,cPRICE,risk, prValue); 

   
       
return 0;
}



int init(){



for(int j= 0; j<maxTP; j++)//loop and read the prediction for every single product
{

   int file_handle = FileOpen(Prediction_Directory+"//"+TP_Code[j]+".csv",FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV);
   string temp = FileReadString(file_handle);
   Print("Close Prediction ",TP_Code[j],"    ", temp);
   
   Predicted_value[j] = StrToDouble(temp);
  
   

}


 for(int i = 0; i<maxTP; i++)
   //go through every single product
   {
   double temp[21];
   read_data(TP_Code[i], temp);
   
   for(int j =0; j<21; j++)
      { 
      all_NQPL[i][j] = temp[j];
      //Print("for product ",TP_Code[i],"the normalized energy level ",j," is ",temp[j]);
      //Print("NQPL: " , TP_Code[i], "   ", temp[j]);
      }
      //quantumTrader(temp,TP_Code[i]);
   }
      
   //while (1){
   for(int i = 0; i<maxTP; i++)//loop through all the products
   { 
      double temp[21];
      
     for(int j = 0; j<21; j++){
      temp[j] = all_NQPL[i][j];
          //place NQPL for a given product into an array
      //Print("NQPL: " , TP_Code[i], "   ", temp[j]);
      
      }
     quantumTrader(temp,TP_Code[i],Predicted_value[i]);   //access the trading function for the given product
   }
//}

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
