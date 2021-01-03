//
//  ArrayVectorExtention.swift
//  ArticleClassifierCore-macos
//
//  Created by Jonathan Duss on 03.03.19.
//

import Foundation

fileprivate extension Numeric {
    func toDouble() -> Double {
        switch self {
        case let n as Double:
            return n
        case let n as Int:
            return Double(n)
        case let n as Float:
            return Double(n)
        case let n as Float64:
            return Double(n)
        case let n as Int8:
            return Double(n)
        case let n as Int16:
            return Double(n)
        case let n as Int32:
            return Double(n)
        case let n as Int64:
            return Double(n)
        default:
            fatalError("This type is not supported \(type(of: self))")
        }
    }
}

extension Array where Element: Numeric {
    
    func HadamarProduct<E>(secondArray: Array<E>) -> [Double] where E: Numeric{
        guard self.count == secondArray.count else {
            fatalError("The Hadamar product require both array to be the same size.")
        }
        var result = Array<Double>(repeating: 0, count: self.count)
        for  idx in 0..<self.count {
            
            result[idx] = self[idx].toDouble() * secondArray[idx].toDouble()
        }
        
        return result
    }
}

extension ContiguousArray where Element: Numeric {
    
    func HadamarProduct<E>(secondArray: ContiguousArray<E>) -> ContiguousArray<Double> where E: Numeric{
        guard self.count == secondArray.count else {
            fatalError("The Hadamar product require both array to be the same size.")
        }
        var result = ContiguousArray<Double>(repeating: 0, count: self.count)
        for  idx in 0..<self.count {
            
            result[idx] = self[idx].toDouble() * secondArray[idx].toDouble()
        }
        
        return result
    }
}
