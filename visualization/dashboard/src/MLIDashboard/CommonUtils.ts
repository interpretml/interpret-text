export class Utils {
  public static argsort (toSort: number[]): number[] {
    const sorted = toSort.map((val, index) => [val, index])
    sorted.sort((a, b) => {
      return a[0] - b[0]
    })
    return sorted.map(val => val[1])
  }

  public static sortedTopK(list:number[], k:number, posOnly:boolean, negOnly:boolean): number[]{
    let sortedList: number[]
    if ((posOnly && negOnly) || (!posOnly && !negOnly)){
      sortedList = this.argsort(list.map(Math.abs)).reverse().splice(0, k)
    } 
    else if (negOnly){
      sortedList = this.argsort(list).splice(0, k)
    }
    else {
      sortedList = this.argsort(list).reverse().splice(0, k)
    }
    return sortedList
  }
}