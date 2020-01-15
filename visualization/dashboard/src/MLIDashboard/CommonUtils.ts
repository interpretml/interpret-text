export class Utils {
  public static argsort (toSort: number[]): number[] {
    const sorted = toSort.map((val, index) => [val, index])
    sorted.sort((a, b) => {
      return a[0] - b[0]
    })
    return sorted.map(val => val[1])
  }

  public static sortedTopK(list:number[], k:number, radio:string): number[]{
    let sortedList: number[]
    if (radio == "all"){
      sortedList = this.argsort(list.map(Math.abs)).reverse().splice(0, k).reverse()
    } 
    else if (radio == "neg"){
      sortedList = this.argsort(list).splice(0, k).reverse()
    }
    else if (radio == "pos") {
      sortedList = this.argsort(list).reverse().splice(0, k).reverse()
    }
    return sortedList
  }
}
