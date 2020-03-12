package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"strconv"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
)

func main() {
	f, err := os.Open("data/train.csv")
	if err != nil {
		log.Fatal(err)
	}

	hdr, data, indices, err := ingest(f)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Original Data: \nRows: %d, Cols: %d\n========\n", len(data), len(hdr))
	c := cardinality(indices)
	for i, h := range hdr {
		fmt.Printf("%v: %v\n", h, c[i])
	}
	fmt.Println("")

	fmt.Printf("Building into matrices\n=============\n")
	rows, cols, XsBack, YsBack, _, _ := clean(hdr, data, indices, datahints, nil)
	Xs := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(XsBack))
	Ys := tensor.New(tensor.WithShape(rows, 1), tensor.WithBacking(YsBack))
	fmt.Printf("Xs:\n%+1.1s\nYs:\n%1.1s\n", Xs, Ys)
	fmt.Println("")

	ofInterest := 19 // variable of interest is in column 19
	cef := CEF(YsBack, ofInterest, indices)

	plt, err := plotCEF(cef)
	if err != nil {
		log.Fatal(err)
	}

	plt.Title.Text = fmt.Sprintf("CEF for %v", hdr[ofInterest])
	plt.X.Label.Text = hdr[ofInterest]
	plt.Y.Label.Text = "Conditionally Expected House Price"

	err = plt.Save(25*vg.Centimeter, 25*vg.Centimeter, "CEF.png")
	if err != nil {
		log.Fatal(err)
	}

	hist, err := plotHist(YsBack)
	if err != nil {
		log.Fatal(err)
	}

	hist.Title.Text = "Histogram of House Prices"
	err = hist.Save(25*vg.Centimeter, 25*vg.Centimeter, "hist.png")
	if err != nil {
		log.Fatal(err)
	}

	for i := range YsBack {
		YsBack[i] = math.Log1p(YsBack[i])
	}
	hist2, err := plotHist(YsBack)
	if err != nil {
		log.Fatal(err)
	}

	hist2.Title.Text = "Histogram of House Prices (Processed)"
	err = hist2.Save(25*vg.Centimeter, 25*vg.Centimeter, "hist2.png")
	if err != nil {
		log.Fatal(err)
	}

	it, err := native.MatrixF64(Xs)
	if err != nil {
		log.Fatal(err)
	}

	for i, isCat := range datahints {
		if isCat {
			continue
		}
		skewness := skew(it, i)
		if skewness > 0.75 {
			log1pCol(it, i)
		}
	}
}

// ingest is a function that ingests the file and outputs the header, data, and index.
func ingest(f io.Reader) (header []string, data [][]string, indices []map[string][]int, err error) {
	r := csv.NewReader(f)

	// handle header
	if header, err = r.Read(); err != nil {
		return
	}

	indices = make([]map[string][]int, len(header))
	var rowCount, colCount int = 0, len(header)
	for rec, err := r.Read(); err == nil; rec, err = r.Read() {
		if len(rec) != colCount {
			return nil, nil, nil, fmt.Errorf("Expected Columns: %d. Got %d columns in row %d", colCount, len(rec), rowCount)
		}
		data = append(data, rec)
		for j, val := range rec {
			if indices[j] == nil {
				indices[j] = make(map[string][]int)
			}
			indices[j][val] = append(indices[j][val], rowCount)
		}
		rowCount++
	}
	return
}

// cardinality counts the number of unique values in a column.
// This assumes that the index i of indices represents a column.
func cardinality(indices []map[string][]int) []int {
	retVal := make([]int, len(indices))
	for i, m := range indices {
		retVal[i] = len(m)
	}
	return retVal
}

// hints is a slice of bools indicating whether it's a categorical variable
func clean(hdr []string, data [][]string, indices []map[string][]int, hints []bool, ignored []string) (int, int, []float64, []float64, []string, []bool) {
	modes := mode(indices)
	var Xs, Ys []float64
	var newHints []bool
	var newHdr []string
	var cols int

	for i, row := range data {

		for j, col := range row {
			if hdr[j] == "Id" { // skip id
				continue
			}
			if hdr[j] == "SalePrice" { // we'll put SalePrice into Ys
				cxx, _ := convert(col, false, nil, hdr[j])
				Ys = append(Ys, cxx...)
				continue
			}

			if inList(hdr[j], ignored) {
				continue
			}

			if hints[j] {
				col = imputeCategorical(col, j, hdr, modes)
			}
			cxx, newHdrs := convert(col, hints[j], indices[j], hdr[j])
			Xs = append(Xs, cxx...)

			if i == 0 {
				h := make([]bool, len(cxx))
				for k := range h {
					h[k] = hints[j]
				}
				newHints = append(newHints, h...)
				newHdr = append(newHdr, newHdrs...)
			}
		}
		// add bias

		if i == 0 {
			cols = len(Xs)
		}
	}
	rows := len(data)
	if len(Ys) == 0 { // it's possible that there are no Ys (i.e. the test.csv file)
		Ys = make([]float64, len(data))
	}
	return rows, cols, Xs, Ys, newHdr, newHints
}

// imputeCategorical replaces "NA" with the mode of categorical values
func imputeCategorical(a string, col int, hdr []string, modes []string) string {
	if a == "NA" || a == "" {
		switch hdr[col] {
		case "MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType", "Exterior1st", "Exterior2nd":
			return modes[col]
		}
	}

	return a
}

// convert converts a string into a slice of floats
func convert(a string, isCat bool, index map[string][]int, varName string) ([]float64, []string) {
	if isCat {
		return convertCategorical(a, index, varName)
	}
	// here we deliberately ignore errors, because the zero value of float64 is well, zero.
	f, _ := strconv.ParseFloat(a, 64)
	return []float64{f}, []string{varName}
}

// convertCategorical is a basic function that encodes a categorical variable as a slice of floats.
// There are no smarts involved at the moment.
// The encoder takes the first value of the map as the default value, encoding it as a []float{0,0,0,...}
func convertCategorical(a string, index map[string][]int, varName string) ([]float64, []string) {
	retVal := make([]float64, len(index)-1)

	// important: Go actually randomizes access to maps, so we actually need to sort the keys
	// optimization point: this function can be made stateful.
	tmp := make([]string, 0, len(index))
	for k := range index {
		tmp = append(tmp, k)
	}

	// numerical "categories" should be sorted numerically
	tmp = tryNumCat(a, index, tmp)

	// find NAs and swap with 0
	var naIndex int
	for i, v := range tmp {
		if v == "NA" {
			naIndex = i
			break
		}
	}
	tmp[0], tmp[naIndex] = tmp[naIndex], tmp[0]

	// build the encoding
	for i, v := range tmp[1:] {
		if v == a {
			retVal[i] = 1
			break
		}
	}
	for i, v := range tmp {
		tmp[i] = fmt.Sprintf("%v_%v", varName, v)
	}

	return retVal, tmp[1:]
}

// mode finds the most common value for each variable
func mode(index []map[string][]int) []string {
	retVal := make([]string, len(index))

	for i, m := range index {
		var max int
		for k, v := range m {
			if len(v) > max {
				max = len(v)
				retVal[i] = k
			}
		}
	}

	return retVal
}

func CEF(Ys []float64, col int, index []map[string][]int) map[string]float64 {
	retVal := make(map[string]float64)
	for k, v := range index[col] {
		var mean float64
		for _, i := range v {
			mean += Ys[i]
		}
		mean /= float64(len(v))
		retVal[k] = mean
	}
	return retVal
}

// plotCEF plots the CEF. This is a simple plot with only the CEF.
// More advanced plots can be also drawn to expose more nuance in understanding the data.
func plotCEF(m map[string]float64) (*plot.Plot, error) {
	ordered := make([]string, 0, len(m))
	for k := range m {
		ordered = append(ordered, k)
	}
	sort.Strings(ordered)

	p, err := plot.New()
	if err != nil {
		return nil, err
	}

	points := make(plotter.XYs, len(ordered))
	for i, val := range ordered {
		// if val can be converted into a float, we'll use it
		// otherwise, we'll stick with using the index
		points[i].X = float64(i)
		if x, err := strconv.ParseFloat(val, 64); err == nil {
			points[i].X = x
		}

		points[i].Y = m[val]
	}
	if err := plotutil.AddLinePoints(p, "CEF", points); err != nil {
		return nil, err
	}
	return p, nil
}

func tryNumCat(a string, index map[string][]int, catStrs []string) []string {
	isNumCat := true
	cats := make([]int, 0, len(index))
	for k := range index {
		i64, err := strconv.ParseInt(k, 10, 64)
		if err != nil && k != "NA" {
			isNumCat = false
			break
		}
		cats = append(cats, int(i64))
	}

	if isNumCat {
		sort.Ints(cats)
		for i := range cats {
			catStrs[i] = strconv.Itoa(cats[i])
		}
		if _, ok := index["NA"]; ok {
			catStrs[0] = "NA" // there are no negative numerical categories
		}
	} else {
		sort.Strings(catStrs)
	}
	return catStrs
}

func inList(a string, l []string) bool {
	for _, v := range l {
		if a == v {
			return true
		}
	}
	return false
}

// skew returns the skewness of a column/variable
func skew(it [][]float64, col int) float64 {
	a := make([]float64, 0, len(it[0]))
	for _, row := range it {
		for _, col := range row {
			a = append(a, col)
		}
	}
	return stat.Skew(a, nil)
}

// log1pCol applies the log1p transformation on a column
func log1pCol(it [][]float64, col int) {
	for i := range it {
		it[i][col] = math.Log1p(it[i][col])
	}
}

// plotHist plots the histogram of a slice of float64s.
func plotHist(a []float64) (*plot.Plot, error) {
	h, err := plotter.NewHist(plotter.Values(a), 10)
	if err != nil {
		return nil, err
	}
	p, err := plot.New()
	if err != nil {
		return nil, err
	}

	h.Normalize(1)
	p.Add(h)
	return p, nil
}
