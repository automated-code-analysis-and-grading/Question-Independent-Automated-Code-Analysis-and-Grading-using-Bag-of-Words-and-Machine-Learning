#include <stdio.h>

long long nCr(int n,int r)
{
  int  i=0;
  long long result=1;
  if (r > n/2)
  {
    r = n - r;
  }
  for (i = 0; i < r; i++)
  {
    result *= (n-i);
    result /= (i+1);
  }
  return result;
}

int main(void)
{
  int t,n,r,i;

  scanf("%d",&t);
  for(i=0;i<t;i++)
  {
    scanf("%d%d",&n,&r);
    printf("%lld\n",nCr(n-1,r-1));//no. of integral solution of x1+x2+x3+......+xr=n,where xi>=1, is C(n-1,r-1)
  }
  return 0;
}
