#include<stdio.h>
int main()
{
    int t;
    scanf("%d",&t);
    while(t--)
    {
       long long int n,k,a,i;
       scanf("%lld%lld",&n,&k);
       a=1ll;
       n--;
       k--;
       if(k>n/2)
        k=n-k;
       for(i=1;i<=k;i++)
       {
         a=a*(n)/i;
          n--;
        }
        printf("%lld\n",a);
      }
     return 0;
     }
