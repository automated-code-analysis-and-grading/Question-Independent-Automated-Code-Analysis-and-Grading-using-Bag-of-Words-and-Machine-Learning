#include<stdio.h>
int main(){
int t,i,j,n;
int a[10000];
long long int count=0,p=1;
scanf("%d",&t);
while(t--){
   count=0;
   p=1;
   scanf("%d",&n);
   scanf("%d",&a[0]);
   for(i=1;i<n;i++){
      scanf("%d",&a[i]);
      if(a[i]>a[i-1] || a[i]==a[i-1])
          p++;
      
      else{
          count+=(p*(p+1))/2-p;
          p=1;
      }
   }
   if(i==n)count+=(p*(p+1))/2-p;
   count+=n; //every element is also a sub array so at last adding single element sub arrays
   printf("%lld\n",count);
}
return 0;

}
