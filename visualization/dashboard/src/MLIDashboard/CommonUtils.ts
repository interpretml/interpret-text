export enum RadioKeys {
  /* keys for the radio buttons (also called ChoiceGroup) */
  all = 'all',
  pos = 'pos',
  neg = 'neg'
}
export class Utils {
  public static argsort (toSort: number[]): number[] {
    /* 
    * returns the indicies that would sort an array 
    * param toSort: the list that you want to be sorted 
    */
    const sorted = toSort.map((val, index) => [val, index])
    sorted.sort((a, b) => {
      return a[0] - b[0]
    })
    return sorted.map(val => val[1])
  }

  public static sortedTopK (list:number[], k:number, radio:string): number[] {
    /*
    * returns a list of indices for the tokens to be displayed based on user controls for number of tokens and type of tokens, list will be of len(number of relevant tokens)
    * returns an empty list if there are no tokens that match the radio key
    * param list: the list that needs to be sorted and spliced
    * param k: the maximum length of the returning list
    * param radio: the key which determines which type of feature importance words will be returned
     */
    let sortedList: number[]
    if (radio === RadioKeys.all) {
      sortedList = this.takeTopK(this.argsort(list.map(Math.abs)).reverse(), k)
    } else if (radio === RadioKeys.neg) {
      sortedList = this.takeTopK(this.argsort(list), k)
      if (list[sortedList[sortedList.length - 1]] >= 0) {
        sortedList = []
      } else {
        for (var i = sortedList.length; i > 0; i--) {
          if (list[sortedList[i]] >= 0) {
            sortedList = sortedList.slice(i + 1, sortedList.length)
            break
          }
        }
      }
    } else if (radio === RadioKeys.pos) {
      sortedList = this.takeTopK(this.argsort(list).reverse(), k)
      if (list[sortedList[sortedList.length - 1]] <= 0) {
        sortedList = []
      } else {
        for (var i = sortedList.length; i > 0; i--) {
          if (list[sortedList[i]] <= 0) {
            sortedList = sortedList.slice(i + 1, sortedList.length)
            break
          }
        }
      }
    }
    return sortedList
  }

  public static takeTopK (list:number[], k:number) {
    /* 
    * returns a list after splicing and taking the top K
    * param list: the list to splice
    * param k: the number to splice the list by
    */
    return list.splice(0, k).reverse()
  }

  public static countNonzeros(list: number[]): number {
    /*
    * returns the count of the nonzero numbers in an array
    * param list: the list in which the numbers are looked at
     */
    let counter = 0
    for (const i in list) {
      if (list[i] !== 0) {
        counter++
      }
    }
    return counter
  }
  public static predictClass(className: any[], prediction:number[]):string{
    /*
     * returns the predicted class 
     * param className: the list of possible classes
     *  param prediction: a vector encoding of the probabilities (or one-hot vector) representing the predictiosns for each class
    */
    return className[this.argsort(prediction)[0]]
  }
}
